import json
import boto3
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET

# ---------------------------
# AWS Clients
# ---------------------------
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# ---------------------------
# Utils
# ---------------------------

def normalize_hits(hits):
    """Normalize hit scores relative to the max score"""
    if not hits:
        return []

    max_score = hits[0].get("_score", 1.0) or 1.0
    normalized = []

    for h in hits:
        normalized.append({
            "id": h["_id"],
            "raw_score": h["_score"],
            "normalized_score": round(h["_score"] / max_score, 3)
        })

    return normalized


def decide_strategy(score):
    """Decide rewrite strategy based on match score"""
    if score > 0.75:
        return "MINOR_REWRITE"
    elif score > 0.4:
        return "ENHANCE"
    else:
        return "RESTRUCTURE"


def build_resume_prompt(section_name, existing_content, job_description, strategy):
    """Build prompt for Claude to optimize resume section"""
    return f"""
You are an ATS-optimized resume assistant.

STRICT RULES:
- DO NOT invent experience
- DO NOT add companies, tools, or years not present
- ONLY rewrite using existing content
- You MAY rephrase to better align with job description

SECTION: {section_name}
STRATEGY: {strategy}

EXISTING RESUME CONTENT:
{existing_content}

JOB DESCRIPTION:
{job_description}

TASK:
Rewrite the section to better match the job description.
Preserve all factual information.
Output ONLY the rewritten section text.
"""


def call_claude(prompt):
    """Call Claude via Bedrock to optimize resume section"""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=body
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


def get_opensearch_client():
    """Create authenticated OpenSearch client"""
    return OpenSearch(
        hosts=[{
            "host": os.environ["OPENSEARCH_ENDPOINT"],
            "port": 443
        }],
        http_auth=(
            os.environ["OPENSEARCH_USER"],
            os.environ["OPENSEARCH_PASSWORD"]
        ),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )


def safe_json_parse(data):
    """Safely parse JSON string or return dict as-is"""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {}
    return data if isinstance(data, dict) else {}


# ---------------------------
# DOCX Section Replacement Functions
# ---------------------------

# Common resume section headers
SECTION_HEADERS = [
    "EXPERIENCE SUMMARY",
    "PROFESSIONAL SUMMARY",
    "SUMMARY",
    "WORK EXPERIENCE",
    "EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "EDUCATION",
    "ACADEMICS",
    "SKILLS",
    "TECHNICAL SKILLS",
    "CERTIFICATIONS",
    "PROJECTS",
    "ACHIEVEMENTS",
    "AWARDS"
]

# XML namespace for DOCX
WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
W = WORD_NAMESPACE


def extract_text_from_docx(docx_bytes):
    """
    Extract text from DOCX without external libraries
    Returns: List of paragraphs
    """
    paragraphs = []
    
    try:
        with zipfile.ZipFile(BytesIO(docx_bytes)) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            for paragraph in tree.iter(f'{W}p'):
                texts = []
                for text_elem in paragraph.iter(f'{W}t'):
                    if text_elem.text:
                        texts.append(text_elem.text)
                
                para_text = ''.join(texts)
                paragraphs.append(para_text)
    
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return []
    
    return paragraphs


def identify_sections_from_text(paragraphs):
    """
    Identify sections from extracted paragraphs
    Returns: Dict with section names and their line ranges
    """
    sections = {}
    current_section = None
    
    for idx, para_text in enumerate(paragraphs):
        text = para_text.strip().upper()
        
        if not text:
            continue
        
        # Check if this is a section header
        matched_section = None
        for section_name in SECTION_HEADERS:
            if text == section_name or text.startswith(section_name):
                matched_section = section_name
                break
        
        if matched_section:
            # Save previous section
            if current_section:
                sections[current_section]['end_idx'] = idx - 1
            
            # Start new section
            current_section = matched_section
            sections[current_section] = {
                'start_idx': idx,
                'end_idx': None,
                'header_text': para_text,
                'content_lines': []
            }
        elif current_section:
            # Add to current section
            sections[current_section]['content_lines'].append(para_text)
    
    # Close last section
    if current_section:
        sections[current_section]['end_idx'] = len(paragraphs) - 1
    
    return sections


def replace_docx_sections_xml(docx_bytes, section_updates):
    """
    Replace sections in DOCX by manipulating XML directly
    
    Args:
        docx_bytes: Original resume as bytes
        section_updates: Dict mapping section names to new content
    
    Returns:
        Modified resume as bytes
    """
    try:
        input_zip = BytesIO(docx_bytes)
        output_zip = BytesIO()
        
        with zipfile.ZipFile(input_zip, 'r') as docx_in:
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as docx_out:
                
                # Read document.xml
                xml_content = docx_in.read('word/document.xml')
                tree = ET.fromstring(xml_content)
                
                # Get body element
                body = tree.find(f'{W}body')
                if body is None:
                    raise ValueError("Could not find document body")
                
                # Extract paragraphs
                paragraphs = []
                para_elements = list(body.findall(f'{W}p'))
                
                for para in para_elements:
                    texts = [t.text for t in para.iter(f'{W}t') if t.text]
                    para_text = ''.join(texts)
                    paragraphs.append(para_text)
                
                # Identify sections
                sections = identify_sections_from_text(paragraphs)
                print(f"Identified sections: {list(sections.keys())}")
                
                # Replace section content
                for section_name, new_content in section_updates.items():
                    section_name_upper = section_name.upper()
                    
                    if section_name_upper not in sections:
                        print(f"Warning: Section '{section_name}' not found")
                        continue
                    
                    print(f"Replacing section: {section_name_upper}")
                    section_info = sections[section_name_upper]
                    start_idx = section_info['start_idx'] + 1  # After header
                    end_idx = section_info['end_idx']
                    
                    if end_idx is None:
                        end_idx = len(paragraphs) - 1
                    
                    # Get the header paragraph element for reference formatting
                    header_para = para_elements[section_info['start_idx']]
                    
                    # Get first content paragraph for formatting reference
                    first_content_para = None
                    if start_idx < len(para_elements):
                        first_content_para = para_elements[start_idx]
                    
                    # Remove old content paragraphs
                    for idx in range(start_idx, min(end_idx + 1, len(para_elements))):
                        if idx < len(para_elements):
                            para_to_remove = para_elements[idx]
                            body.remove(para_to_remove)
                    
                    # Insert new content after header
                    header_idx = list(body).index(header_para)
                    new_lines = new_content.strip().split('\n')
                    
                    for i, line in enumerate(new_lines):
                        if not line.strip():
                            continue
                        
                        # Create new paragraph with formatting
                        new_para = create_paragraph_element(line, first_content_para)
                        body.insert(header_idx + 1 + i, new_para)
                
                # Write modified document.xml
                modified_xml = ET.tostring(tree, encoding='unicode')
                docx_out.writestr('word/document.xml', modified_xml)
                
                # Copy all other files unchanged
                for item in docx_in.infolist():
                    if item.filename != 'word/document.xml':
                        docx_out.writestr(item, docx_in.read(item.filename))
        
        output_zip.seek(0)
        return output_zip.read()
    
    except Exception as e:
        print(f"Error replacing sections: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_paragraph_element(text, reference_para=None):
    """
    Create a new paragraph element with text, optionally copying formatting
    
    Args:
        text: Text content
        reference_para: Optional paragraph to copy formatting from
    
    Returns:
        XML Element for paragraph
    """
    para = ET.Element(f'{W}p')
    
    # Copy paragraph properties if reference exists
    if reference_para is not None:
        ref_pPr = reference_para.find(f'{W}pPr')
        if ref_pPr is not None:
            para.append(ref_pPr)
    
    # Create run
    run = ET.SubElement(para, f'{W}r')
    
    # Copy run properties if reference exists
    if reference_para is not None:
        ref_runs = reference_para.findall(f'{W}r')
        if ref_runs:
            ref_rPr = ref_runs[0].find(f'{W}rPr')
            if ref_rPr is not None:
                run.append(ref_rPr)
    
    # Add text
    text_elem = ET.SubElement(run, f'{W}t')
    text_elem.text = text
    text_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    
    return para


def download_resume_from_s3(bucket, key):
    """
    Download resume from S3 as binary
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
    
    Returns:
        Resume content as bytes
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


def upload_resume_to_s3(bucket, key, content_bytes, content_type=None):
    """
    Upload enhanced resume to S3
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        content_bytes: Resume content as bytes
        content_type: MIME type (optional)
    """
    if content_type is None:
        content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content_bytes,
        ContentType=content_type
    )
    print(f"Uploaded enhanced resume to: s3://{bucket}/{key}")


def generate_enhanced_resume(resume_key, bucket, updated_sections):
    """
    Generate enhanced DOCX resume with replaced sections
    
    Args:
        resume_key: Original resume S3 key
        bucket: S3 bucket name
        updated_sections: Dict of section names to updated content
    
    Returns:
        Enhanced resume bytes and S3 key where it was saved
    """
    try:
        # Download original resume
        print(f"Downloading resume from S3: {resume_key}")
        original_resume_bytes = download_resume_from_s3(bucket, resume_key)
        
        # Verify it's a DOCX file
        if not resume_key.lower().endswith('.docx'):
            raise ValueError(f"Only DOCX files are supported. Received: {resume_key}")
        
        # Build section updates dict (extract just the updated_content)
        section_updates = {}
        for section_name, content_dict in updated_sections.items():
            section_updates[section_name] = content_dict['updated_content']
        
        print(f"Replacing {len(section_updates)} sections")
        
        # Replace sections using XML manipulation
        enhanced_bytes = replace_docx_sections_xml(original_resume_bytes, section_updates)
        
        # Generate output key
        base_name = resume_key.rsplit('.', 1)[0]
        output_key = f"{base_name}_enhanced.docx"
        
        # Upload enhanced resume
        upload_resume_to_s3(bucket, output_key, enhanced_bytes)
        
        return enhanced_bytes, output_key
        
    except Exception as e:
        print(f"Error generating enhanced resume: {e}")
        import traceback
        traceback.print_exc()
        raise


# ---------------------------
# Lambda Handler
# ---------------------------

def lambda_handler(event, context):
    """
    Main handler to optimize resume sections based on job description match
    Works with DOCX files only
    """
    try:
        body = json.loads(event["body"])
        resume_key = body["resume_key"]
        bucket = body.get("bucket", os.environ.get("RESUME_BUCKET", ""))
        
        # Validate DOCX format
        if not resume_key.lower().endswith('.docx'):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Only DOCX format is supported",
                    "message": f"Please upload a DOCX file. Received: {resume_key}"
                })
            }
        
        # Option to skip resume generation (for testing)
        generate_resume = body.get("generate_resume", True)

        opensearch = get_opensearch_client()

        # Query OpenSearch for all sections of this resume
        response = opensearch.search(
            index="resume-index-v3",
            body={
                "query": {
                    "match": {
                        "resume_key": resume_key
                    }
                },
                "size": 100
            }
        )

        hits = response["hits"]["hits"]
        if not hits:
            return {
                "statusCode": 404,
                "body": json.dumps({"message": "Resume not found in index"})
            }

        updated_resume = {}

        for hit in hits:
            try:
                doc_id = hit["_id"]
                source = hit["_source"]
                
                print(f"Processing doc_id: {doc_id}")

                # Extract section name
                section = doc_id.split("#")[-1].replace("_", " ")
                print(f"Section: {section}")

                # Get existing resume content
                existing_content = source.get("resume_sections", "None")
                print(f"Existing content for {section}: {existing_content[:100]}...")

                # Get job requirements
                requirements = source.get("requirements", "{}")
                requirements_dict = safe_json_parse(requirements)
                
                if requirements_dict and requirements_dict != {}:
                    requirements_text = json.dumps(requirements_dict, indent=2)
                else:
                    requirements_text = requirements
                
                print(f"Requirements type: {type(requirements_text)}")

                # Get similarity score
                similarity_score_raw = source.get("similarity_score_raw", "{}")
                similarity_dict = safe_json_parse(similarity_score_raw)
                
                print(f"Similarity score dict: {similarity_dict}")

                # Extract hits from similarity score
                similarity_hits = []
                if "hits" in similarity_dict:
                    if isinstance(similarity_dict["hits"], dict):
                        similarity_hits = similarity_dict["hits"].get("hits", [])
                    elif isinstance(similarity_dict["hits"], list):
                        similarity_hits = similarity_dict["hits"]
                
                print(f"Similarity hits: {similarity_hits}")

                # Normalize scores
                normalized_hits = normalize_hits(similarity_hits)
                print(f"Normalized hits: {normalized_hits}")

                # Find score for THIS section
                section_score = 0.0
                section_id_pattern = section.replace(" ", "_")
                
                for s in normalized_hits:
                    if s["id"].endswith(section_id_pattern):
                        section_score = s["normalized_score"]
                        break
                
                print(f"Section score for {section}: {section_score}")

                # Decide optimization strategy
                strategy = decide_strategy(section_score)
                print(f"Strategy for {section}: {strategy}")

                # Build prompt and call Claude
                prompt = build_resume_prompt(
                    section_name=section,
                    existing_content=existing_content,
                    job_description=requirements_text,
                    strategy=strategy
                )

                updated_text = call_claude(prompt)

                # Store updated section
                updated_resume[section] = {
                    "match_score": section_score,
                    "strategy": strategy,
                    "original_content": existing_content,
                    "updated_content": updated_text
                }

                print(f"Updated section {section} successfully")

            except Exception as e:
                print(f"Error processing section {section}: {e}")
                continue

        # Create formatted output
        updated_resume_str = ""
        for section, content in updated_resume.items():
            updated_resume_str += f"\n{'='*60}\n"
            updated_resume_str += f"{section.upper()}\n"
            updated_resume_str += f"{'='*60}\n"
            updated_resume_str += f"Match Score: {content['match_score']}\n"
            updated_resume_str += f"Strategy: {content['strategy']}\n\n"
            updated_resume_str += f"{content['updated_content']}\n"

        print(f"Complete updated resume:\n{updated_resume_str}")

        # Generate enhanced DOCX resume
        enhanced_resume_key = None
        if generate_resume and bucket:
            try:
                print("\n--- Generating Enhanced DOCX Resume ---")
                enhanced_bytes, enhanced_resume_key = generate_enhanced_resume(
                    resume_key=resume_key,
                    bucket=bucket,
                    updated_sections=updated_resume
                )
                print(f"✓ Enhanced resume saved to: {enhanced_resume_key}")
            except Exception as e:
                print(f"Warning: Could not generate enhanced resume file: {e}")
                # Continue without failing the entire request

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Resume optimized successfully",
                "resume_key": resume_key,
                "enhanced_resume_key": enhanced_resume_key,
                "sections_updated": len(updated_resume),
                "updated_resume": updated_resume,
                "formatted_resume": updated_resume_str
            })
        }

    except Exception as e:
        print(f"Lambda handler error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "message": "Failed to optimize resume"
            })
        }
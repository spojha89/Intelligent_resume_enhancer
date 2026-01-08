import json
import boto3
import io
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
import re
import zipfile
import xml.etree.ElementTree as ET

s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# XML namespace for DOCX
WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
W = WORD_NAMESPACE


def safe_json_loads(text: str):
    """
    Extract and parse JSON from LLM output safely with multiple fallback strategies
    """
    # Remove any markdown code blocks
    text = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse failed: {e}")
        print(f"Problematic text: {text[:500]}...")
        
        # Try to extract JSON object using regex (greedy match)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e2:
                print(f"Regex extraction failed: {e2}")
        
        # Try to fix common JSON issues
        try:
            # Replace single quotes with double quotes
            fixed_text = text.replace("'", '"')
            # Remove trailing commas before closing braces/brackets
            fixed_text = re.sub(r',\s*([}\]])', r'\1', fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError as e3:
            print(f"JSON fixing attempt failed: {e3}")
            
        raise ValueError(f"Claude output is not valid JSON. Original error: {e}")


def section_to_text(section_content):
    """Convert section content to text"""
    if isinstance(section_content, str):
        return section_content
    if isinstance(section_content, dict):
        return " | ".join(
            f"{k}: {v}" for k, v in section_content.items()
        )
    if isinstance(section_content, list):
        return " | ".join(map(str, section_content))
    return str(section_content)


def get_opensearch_client():
    """Create authenticated OpenSearch client"""
    USER = os.environ.get("OPENSEARCH_USER")
    PASSWORD = os.environ.get("OPENSEARCH_PASSWORD")
    awsauth = (USER, PASSWORD)
    OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT")

    return OpenSearch(
        hosts=[{'host': OPENSEARCH_ENDPOINT, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )


def extract_text_from_docx(bucket, key):
    """
    Extract text from DOCX stored in S3
    Uses built-in libraries only (no external dependencies)
    """
    print("Extracting text from DOCX...")
    print("Bucket:", bucket)
    print("Key:", key)
    
    # Download DOCX from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    docx_content = response['Body'].read()
    
    # Extract text from DOCX
    paragraphs = []
    
    try:
        # DOCX is a ZIP file containing XML
        with zipfile.ZipFile(io.BytesIO(docx_content)) as docx:
            # Read the main document XML
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # Extract text from all paragraphs
            for paragraph in tree.iter(f'{W}p'):
                texts = []
                for text_elem in paragraph.iter(f'{W}t'):
                    if text_elem.text:
                        texts.append(text_elem.text)
                
                para_text = ''.join(texts)
                if para_text.strip():  # Only add non-empty paragraphs
                    paragraphs.append(para_text)
        
        # Join all paragraphs with newlines
        full_text = '\n'.join(paragraphs)
        print(f"Extracted {len(paragraphs)} paragraphs from DOCX")
        return full_text
    
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        raise


def extract_text_from_s3(bucket, key):
    """Extract text from text file in S3"""
    print("Extracting text from S3...")
    print("Bucket:", bucket)
    print("Key:", key)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')


def generate_embedding(text):
    """Generate embeddings using Amazon Bedrock Titan"""
    print("Generating embeddings...")
    body = json.dumps({
        "inputText": text
    })
    
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body,
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    print("Embedding created successfully")
    return response_body['embedding']


def extract_requirements_with_claude(job_description, max_retries=2):
    """Use Claude to extract structured requirements from job description with retry logic"""
    print("Extracting requirements with Claude...")

    prompt = f"""Analyze this job description and extract the following information. 
Return ONLY a valid JSON object with these exact keys, nothing else:

{{
  "required_skills": ["skill1", "skill2"],
  "years_of_experience": "X years",
  "key_responsibilities": ["responsibility1", "responsibility2"],
  "required_qualifications": ["qualification1", "qualification2"],
  "preferred_qualifications": ["qualification1", "qualification2"],
  "ats_keywords": ["keyword1", "keyword2"]
}}

Rules:
- Return ONLY the JSON object
- Use "None" as the value if a section is missing
- Do not include any explanations or markdown
- Ensure all strings are properly quoted
- Do not include trailing commas

Job Description:
{job_description}"""

    for attempt in range(max_retries):
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0  # More deterministic output
            })
            
            response = bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            requirements_text = response_body['content'][0]['text']
            
            # Validate JSON parsing
            requirements_json = safe_json_loads(requirements_text)
            
            # Return as JSON string for storage
            return json.dumps(requirements_json)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("All retry attempts exhausted")
                raise
    
    raise Exception("Failed to extract requirements after retries")


def extract_sections_from_resume(resume_text, max_retries=2):
    """Extract sections from resume text with retry logic"""
    print("Extracting sections from resume...")

    prompt = f"""Extract information from this resume and return ONLY a valid JSON object with these exact keys:

{{
  "Experience Summary": "brief summary here",
  "Education": "education details here",
  "Work Experience": "detailed work experience here",
  "Skills": "skills list here",
  "Certifications": "certifications here",
  "Projects": "projects here"
}}

Rules:
- Return ONLY the JSON object
- Use "None" as the value if a section is missing
- For Work Experience, include a few lines about major experiences
- Do not include any explanations, markdown, or code blocks
- Ensure all strings are properly quoted
- Do not include trailing commas

Resume:
{resume_text}"""

    for attempt in range(max_retries):
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0  # More deterministic output
            })

            response = bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                body=body
            )

            response_body = json.loads(response['body'].read())
            sections_text = response_body['content'][0]['text']
            
            # Validate JSON parsing
            sections_json = safe_json_loads(sections_text)
            
            # Return as JSON string for storage
            return json.dumps(sections_json)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("All retry attempts exhausted")
                raise
    
    raise Exception("Failed to extract sections after retries")


def index_document(opensearch_client, index_name, document, doc_id):
    """Index document in OpenSearch"""
    print(f"Indexing document with ID: {doc_id}")
    try:
        response = opensearch_client.index(index=index_name, body=document, id=doc_id)
        print(f"Successfully indexed document: {response['result']}")
        return response

    except Exception as e:
        print(f"Error indexing document {doc_id}: {e}")
        raise  # Re-raise to catch in main handler


def find_similarity_score(opensearch_client, index_name, embedding):
    """Find similarity score in OpenSearch"""
    print("Finding similarity score...")
    try:
        query = {
            "size": 5,
            "_source": False,
            "query": {
                "knn": {
                    "resume_sections_embedding": {
                        "vector": embedding,
                        "k": 5
                    }
                }
            }
        }
        response = opensearch_client.search(index=index_name, body=query)
        print("Response:", response)
        print("Response max score", response["hits"]["max_score"])
        return response

    except Exception as e:
        print("Error finding similarity score:", e)
        return None


def lambda_handler(event, context):
    """
    Extract requirements from job description and generate embeddings
    Supports DOCX resume files
    """
    try:
        body = json.loads(event['body'])
        # Get S3 keys from event
        bucket = body['bucket']
        resume_key = body['resume_key']
        job_desc_key = body['job_desc_key']
        
        # Validate file format
        if not resume_key.lower().endswith('.docx'):
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Only DOCX format is supported for resumes',
                    'message': f'Please upload a DOCX file. Received: {resume_key}'
                })
            }
        
        # Extract text from DOCX resume
        print(f"Processing DOCX resume: {resume_key}")
        resume_text = extract_text_from_docx(bucket, resume_key)
        print(f"Extracted resume text length: {len(resume_text)} characters")
        
        # Extract job description text
        job_desc_text = extract_text_from_s3(bucket, job_desc_key)
        
        # Extract structured requirements using Claude (with retry)
        requirements = extract_requirements_with_claude(job_desc_text)

        # Extract sections from resume (with retry)
        resume_sections = extract_sections_from_resume(resume_text)
        
        # Generate embeddings for job description
        job_embedding = generate_embedding(job_desc_text)
        
        # Generate embeddings for resume sections (for initial indexing)
        resume_embedding = generate_embedding(resume_text)

        # Generate embeddings for resume sections (for initial indexing)
        resume_sections_embedding = generate_embedding(resume_sections)

        # Initialize OpenSearch client
        opensearch_client = get_opensearch_client()

        # Create index if it doesn't exist
        try:
            index_name = 'resume-index-v3'
            # Define index mapping  
            index_body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "resume_embedding": {
                            "type": "knn_vector",
                            "dimension": 1536
                        },
                        "resume_sections_embedding": {
                            "type": "knn_vector",
                            "dimension": 1536
                        },
                        "job_embedding": {
                            "type": "knn_vector",
                            "dimension": 1536
                        },
                        "requirements": {
                            "type": "text"
                        },
                        "similarity_score_raw": {
                            "type": "text"
                        },
                        "bucket": {
                            "type": "keyword"
                        },
                        "resume_sections": {
                            "type": "text"
                        },
                        "resume_key": {
                            "type": "keyword"
                        }
                    }
                }
            }
            # Create index if it doesn't exist
            if not opensearch_client.indices.exists(index=index_name):
                opensearch_client.indices.create(
                    index=index_name,
                    body=index_body
                )
                print(f"Created new index: {index_name}")
            else:
                print(f"Using existing index: {index_name}")
                
        except Exception as e:
            print("Error creating index:", e)
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": str(e)
                })
            }

        
        # Find similarity_score comparing job description and resume
        similarity_score_raw = find_similarity_score(opensearch_client, index_name, job_embedding)
        print("Similarity score raw version:", similarity_score_raw)
        
        # Convert similarity score to string for storage
        if similarity_score_raw:
            similarity_score_str = json.dumps({
                "max_score": similarity_score_raw.get("hits", {}).get("max_score"),
                "total_hits": similarity_score_raw.get("hits", {}).get("total", {}).get("value", 0)
            })
        else:
            similarity_score_str = json.dumps({"max_score": None, "total_hits": 0})

        section_headers = [
            "Experience Summary",
            "Education",
            "Work Experience",
            "Skills",
            "Certifications",
            "Projects"
        ]

        # Convert resume_sections from string to dictionary
        resume_sections_dict = safe_json_loads(resume_sections)

        # Loop through each section in resume_sections and create document for each section
        for section in section_headers:
            raw_section = resume_sections_dict.get(section, "None")
            section_text = section_to_text(raw_section)

            section_embedding = generate_embedding(str(section_text))

            document = {
                "resume_sections_embedding": section_embedding,
                "resume_sections": section_text,
                "resume_embedding": resume_embedding,
                "job_embedding": job_embedding,
                "requirements": requirements,
                "similarity_score_raw": similarity_score_str,
                "bucket": bucket,                
                "resume_key": resume_key
            }

            doc_id = f"{resume_key}#{section.replace(' ', '_')}"
            try:
                index_document(opensearch_client, index_name, document, doc_id)
                print(f"Document indexed successfully for section: {section}")
            except Exception as e:
                print(f"Failed to index section {section}: {e}")
                # Continue with other sections even if one fails
                continue
    

        return {
            'statusCode': 200,
            'body': json.dumps({
                'bucket': bucket,
                'resume_key': resume_key,
                'file_format': 'docx',
                'execution_status': 'success',
                'sections_indexed': len(section_headers),
                'message': 'DOCX resume processed and indexed successfully'
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
                "message": "Failed to process DOCX resume"
            })
        }
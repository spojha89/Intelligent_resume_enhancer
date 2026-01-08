import json
import boto3
import base64
from datetime import datetime

s3_client = boto3.client('s3')
BUCKET_NAME = 'suvus3bucket'

def get_content_type(file_name):
    """
    Determine content type based on file extension
    """
    file_name_lower = file_name.lower()
    
    if file_name_lower.endswith('.pdf'):
        return 'application/pdf'
    elif file_name_lower.endswith('.docx'):
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif file_name_lower.endswith('.doc'):
        return 'application/msword'
    elif file_name_lower.endswith('.txt'):
        return 'text/plain'
    elif file_name_lower.endswith('.json'):
        return 'application/json'
    else:
        return 'application/octet-stream'


def lambda_handler(event, context):
    """
    Upload resume and job description to S3
    Expects: Base64 encoded files or presigned URL request
    Supports: PDF, DOCX, DOC, TXT formats
    """
    print(f"Received event: {event}")
    try:
        body = json.loads(event['body'])
        action = body.get('action', 'upload')

        print(f"Received body: {body}")
        print(f"Received action: {action}")
        
        if action == 'get_presigned_url':
            # Generate presigned URLs for direct upload
            file_type = body['file_type']  # 'resume' or 'job_description'
            file_name = body['file_name']
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{file_type}/{timestamp}_{file_name}"
            
            # Auto-detect content type from file name
            content_type = body.get('content_type') or get_content_type(file_name)
            
            presigned_url = s3_client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': BUCKET_NAME,
                    'Key': s3_key,
                    'ContentType': content_type
                },
                ExpiresIn=3600
            )

            print(f"Presigned URL: {presigned_url}")
            print(f"Bucket: {BUCKET_NAME}")
            print(f"Key: {s3_key}")
            print(f"Content-Type: {content_type}")
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'upload_url': presigned_url,
                    's3_key': s3_key,
                    'bucket': BUCKET_NAME,
                    'content_type': content_type
                })
            }
        
        elif action == 'upload':
            # Direct base64 upload
            resume_content = base64.b64decode(body['resume'])
            job_desc_content = base64.b64decode(body['job_description'])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            user_id = body.get('user_id', 'default_user')
            
            # Get file extensions from body or default to pdf/txt
            resume_file_name = body.get('resume_file_name', 'resume.pdf')
            job_desc_file_name = body.get('job_desc_file_name', 'job_desc.txt')
            
            # Extract extension
            resume_ext = resume_file_name.split('.')[-1]
            job_desc_ext = job_desc_file_name.split('.')[-1]
            
            # Build S3 keys with proper extensions
            resume_key = f"resumes/{user_id}/{timestamp}_{resume_file_name}"
            job_desc_key = f"job_descriptions/{user_id}/{timestamp}_{job_desc_file_name}"
            
            # Determine content types
            resume_content_type = get_content_type(resume_file_name)
            job_desc_content_type = get_content_type(job_desc_file_name)
            
            print(f"Uploading resume: {resume_key} (Content-Type: {resume_content_type})")
            print(f"Uploading job description: {job_desc_key} (Content-Type: {job_desc_content_type})")
            
            # Upload resume
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=resume_key,
                Body=resume_content,
                ContentType=resume_content_type
            )
            
            # Upload job description
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=job_desc_key,
                Body=job_desc_content,
                ContentType=job_desc_content_type
            )
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': 'Files uploaded successfully',
                    'resume_key': resume_key,
                    'job_desc_key': job_desc_key,
                    'bucket': BUCKET_NAME,
                    'resume_format': resume_ext,
                    'resume_content_type': resume_content_type
                })
            }
        
        elif action == 'get_file_info':
            # Get information about uploaded file
            s3_key = body['s3_key']
            
            try:
                response = s3_client.head_object(
                    Bucket=BUCKET_NAME,
                    Key=s3_key
                )
                
                return {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({
                        'exists': True,
                        'content_type': response.get('ContentType'),
                        'size': response.get('ContentLength'),
                        'last_modified': response.get('LastModified').isoformat() if response.get('LastModified') else None
                    })
                }
            except s3_client.exceptions.NoSuchKey:
                return {
                    'statusCode': 404,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({
                        'exists': False,
                        'message': 'File not found'
                    })
                }
            
    except KeyError as e:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': f'Missing required field: {str(e)}'
            })
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': str(e)
            })
        }

'''
 {
        "body": "{\"action\":\"get_presigned_url\",\"file_type\":\"description\",\"file_name\":\"Sample_Job_Description.txt\",\"content_type\":\"text/plain\"}"
      }


{
  "body": "{\"action\":\"get_presigned_url\",\"file_type\":\"resume\",\"file_name\":\"Resume_Subhra_Ojha.pdf\",\"content_type\":\"application/pdf\"}"
}

'''
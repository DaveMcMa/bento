import os
import requests
import boto3
import tempfile
from urllib.parse import urljoin
from pathlib import Path
import json

def get_github_files(repo_url, branch='main'):
    """
    Get all files from a GitHub repository using the GitHub API
    """
    # Extract owner, repo, and path from GitHub URL
    parts = repo_url.replace('https://github.com/', '').split('/')
    owner = parts[0]
    repo = parts[1]
    
    # Handle tree/branch/path structure
    if 'tree' in parts:
        tree_idx = parts.index('tree')
        branch = parts[tree_idx + 1]
        path = '/'.join(parts[tree_idx + 2:]) if len(parts) > tree_idx + 2 else ''
    else:
        path = ''
    
    # GitHub API URL for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    if branch != 'main':
        api_url += f"?ref={branch}"
    
    files = []
    
    def fetch_directory_contents(url, current_path=''):
        response = requests.get(url)
        response.raise_for_status()
        
        contents = response.json()
        
        for item in contents:
            if item['type'] == 'file':
                files.append({
                    'name': item['name'],
                    'path': item['path'],
                    'download_url': item['download_url']
                })
            elif item['type'] == 'dir':
                # Recursively fetch subdirectory contents
                subdir_url = item['url']
                if branch != 'main':
                    subdir_url += f"?ref={branch}"
                fetch_directory_contents(subdir_url, item['path'])
    
    fetch_directory_contents(api_url)
    return files

def download_file(url, local_path):
    """
    Download a file from URL to local path
    """
    response = requests.get(url)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with open(local_path, 'wb') as f:
        f.write(response.content)
    
    return local_path

def upload_to_s3(s3_client, local_file_path, bucket_name, s3_key):
    """
    Upload a file to S3 bucket
    """
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        print(f"Successfully uploaded {s3_key} to bucket {bucket_name}")
        return True
    except Exception as e:
        print(f"Error uploading {s3_key}: {str(e)}")
        return False

def main():
    # Configuration
    GITHUB_REPO_URL = "https://github.com/DaveMcMa/bento/tree/main/1.1.11"
    S3_BUCKET_NAME = "rundmc"
    
    # Initialize S3 client with your custom endpoint
    s3 = boto3.client('s3', endpoint_url='http://local-s3-service.ezdata-system.svc.cluster.local:30000')
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        try:
            # Get all files from GitHub repository
            print("Fetching file list from GitHub...")
            files = get_github_files(GITHUB_REPO_URL)
            print(f"Found {len(files)} files in repository")
            
            # Filter for .bento files
            bento_files = [f for f in files if f['name'].endswith('.bento')]
            print(f"Found {len(bento_files)} .bento files")
            
            if not bento_files:
                print("No .bento files found in the repository")
                return
            
            # Download and upload .bento files
            successful_uploads = 0
            
            for file_info in bento_files:
                print(f"\nProcessing: {file_info['path']}")
                
                # Create local file path
                local_file_path = os.path.join(temp_dir, file_info['path'])
                
                try:
                    # Download file
                    print(f"  Downloading from: {file_info['download_url']}")
                    download_file(file_info['download_url'], local_file_path)
                    
                    # Upload to S3 (preserve the relative path as S3 key)
                    s3_key = file_info['path']
                    print(f"  Uploading to S3 as: {s3_key}")
                    
                    if upload_to_s3(s3, local_file_path, S3_BUCKET_NAME, s3_key):
                        successful_uploads += 1
                    
                except Exception as e:
                    print(f"  Error processing {file_info['path']}: {str(e)}")
            
            print(f"\n=== Summary ===")
            print(f"Total .bento files found: {len(bento_files)}")
            print(f"Successfully uploaded: {successful_uploads}")
            print(f"Failed uploads: {len(bento_files) - successful_uploads}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import requests
        import boto3
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install requests boto3")
        exit(1)
    
    main()

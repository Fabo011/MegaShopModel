import joblib
from supabase import create_client, Client
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_ANON_KEY')
supabase: Client = create_client(url, key)
bucket_name = "megashopmodel"

def upload_model_to_supabase(file_path, file_name):
    with open(file_path, "rb") as file:
        response = supabase.storage.from_(bucket_name).upload(file_name, file)
        print(response)
        
        if not (response.status_code == 200 or response.status_code == 201):
            raise Exception(response.text)
        
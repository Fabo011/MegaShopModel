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
        try:
            supabase.storage.from_(bucket_name).upload(file_name, file)
            print('Models uploaded to Supabase!')
        except Exception:
            # Just for preventing the server from crash if the file is already uploaded
            print("Model already uploaded to Supabase")
        
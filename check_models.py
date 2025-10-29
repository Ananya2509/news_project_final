import google.auth
import google.generativeai as genai

credentials, project_id = google.auth.default()
genai.configure(credentials=credentials)

models = genai.list_models()
print("Available models:\n")
for m in models:
    print("-", m.name)

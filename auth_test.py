import google.auth

try:
    credentials, project = google.auth.default()
    print("✅ Authenticated successfully!")
    print("Project ID:", project)
except Exception as e:
    print("❌ Authentication failed:", e)

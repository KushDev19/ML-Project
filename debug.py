import os

# Run this in your actual project directory
project_path = r"D:\Data Science\ML Project"
artifacts_path = os.path.join(project_path, "artifacts")

print(f"Project path exists: {os.path.exists(project_path)}")
print(f"Artifacts path exists: {os.path.exists(artifacts_path)}")

if os.path.exists(artifacts_path):
    files = os.listdir(artifacts_path)
    print(f"Files in artifacts: {files}")
    
    for file in ['model.pkl', 'preprocessor.pkl']:
        file_path = os.path.join(artifacts_path, file)
        exists = os.path.exists(file_path)
        if exists:
            size = os.path.getsize(file_path)
            print(f"✅ {file}: EXISTS ({size} bytes)")
        else:
            print(f"❌ {file}: MISSING")

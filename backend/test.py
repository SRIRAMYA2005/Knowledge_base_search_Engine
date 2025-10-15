import requests

# Use a raw string (r"...") or forward slashes
file_path = r"C:\Users\gvssp\Downloads\Knowledge-base Search Engine_12.pdf"

url = "http://127.0.0.1:8000/summarize-pdf"

# Open the file in binary mode
with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Print JSON response
print(response.json())

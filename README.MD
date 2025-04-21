# Creating the Lambda Layer
Here’s exactly how to zip your `requests` Lambda Layer using Windows **Command Prompt**.

---

### ✅ Step-by-Step (Windows CMD) to Create `requests` Layer

#### 📁 Folder structure you want:
```
requests-layer/
└── python/
    └── requests/
    └── other dependencies...
```

---

### ✅ 1. Create the folder structure

Open Command Prompt and run:

```cmd
mkdir requests-layer
cd requests-layer
mkdir python
```

---

### ✅ 2. Install `requests` into the `python` folder

Still inside `requests-layer`, run:

```cmd
pip install requests -t python
```

---

### ✅ 3. Go one level up and zip it

Go back one level:

```cmd
cd ..
```

Then zip **only the `python/` folder**, so the ZIP file structure is valid for Lambda.

#### If you have PowerShell available (recommended):

```powershell
powershell Compress-Archive -Path requests-layer\python -DestinationPath requests-layer.zip
```

#### If PowerShell isn’t available and you have Git Bash or WSL, you can use `zip` there instead. Otherwise, install a tool like 7-Zip or use Windows File Explorer.

---

### ✅ 4. Check ZIP contents (Important!)

Make sure `requests-layer.zip` contains this:

```
requests-layer.zip
└── python/
    └── requests/
    └── ...
```

> ❗ **If `python/` is *not* at the root** of the ZIP, Lambda will not recognize it as a valid layer.

---

### ✅ 5. Upload to AWS Lambda Layer

Same steps:
- Go to AWS Console → Lambda → Layers
- Click **Create layer**
- Name it something like `requests-layer`
- Upload your `requests-layer.zip`
- Choose compatible runtimes (e.g. Python 3.9)
- Save

Then:
- Go to your Lambda function
- Scroll to **Layers**
- Click **Add a layer**
- Choose the one you just created


---

# Lambda function destination payload

`Success`
```json
{
  "version": "1.0",
  "timestamp": "2025-04-21T15:22:05.123Z",
  "requestContext": {
    "requestId": "f71bfb77-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
    "functionArn": "arn:aws:lambda:REGION:ACCOUNT_ID:function:YourLambdaFunctionName",
    "condition": "Success",
    "approximateInvokeCount": 1
  },
  "requestPayload": {
    // This is the event your Lambda originally received (e.g., the S3 event)
  },
  "responseContext": {
    "statusCode": 200,
    "executedVersion": "$LATEST",
    "functionError": null
  },
  "responsePayload": {
    "statusCode": 200,
    "body": "{\"message\": \"Label detection and article generation complete\", \"article\": \"...\"}"
  }
}
```

`Failed`
```json
{
  "version": "1.0",
  "timestamp": "2025-04-21T15:22:05.123Z",
  "requestContext": {
    "requestId": "f71bfb77-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
    "functionArn": "arn:aws:lambda:REGION:ACCOUNT_ID:function:YourLambdaFunctionName",
    "condition": "RetriesExhausted",
    "approximateInvokeCount": 3
  },
  "requestPayload": {
    // Same as before, the original event
  },
  "responseContext": {
    "statusCode": 200,
    "executedVersion": "$LATEST",
    "functionError": "Handled" // or "Unhandled"
  },
  "responsePayload": {
    "errorMessage": "Something went wrong",
    "errorType": "ExceptionType",
    "stackTrace": [ /* optional */ ]
  }
}
```
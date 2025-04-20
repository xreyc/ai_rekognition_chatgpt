import json
import boto3
import os
import requests

rekognition_client = boto3.client('rekognition')
openai_api_key = os.environ.get('OPENAI_API_KEY')  # Make sure to set this in Lambda environment variables

def detect_labels_from_s3(bucket_name, object_key, max_labels=10, min_confidence=80):
    try:
        response = rekognition_client.detect_labels(
            Image={'S3Object': {'Bucket': bucket_name, 'Name': object_key}},
            MaxLabels=max_labels,
            MinConfidence=min_confidence
        )
        return [
            {"Name": label['Name'], "Confidence": label['Confidence']}
            for label in response['Labels']
        ]
    except Exception as e:
        print(f"Rekognition error: {e}")
        return []

def generate_article_from_labels(labels):
    if not labels:
        return "No recognizable content detected in the images."

    label_names = [label['Name'] for label in labels]
    prompt = (
        f"Write an engaging and informative article based on the following labels: {', '.join(label_names)}. "
        f"Include a creative description and potential context of the scene."
    )

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",  # or gpt-3.5-turbo
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 800
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Failed to generate article."

def lambda_handler(event, context):
    all_labels = []

    for record in event.get('Records', []):
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"New image uploaded: s3://{bucket}/{key}")

        if key.lower().endswith(('.jpg', '.jpeg', '.png')):
            labels = detect_labels_from_s3(bucket, key)
            print(f"Detected labels for {key}: {labels}")
            all_labels.extend(labels)  # Collect all labels

        else:
            print(f"Skipped non-image file: {key}")

    # Generate one article based on all labels detected
    article = generate_article_from_labels(all_labels)
    print(f"Generated article:\n{article}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Label detection and article generation complete',
            'article': article
        })
    }

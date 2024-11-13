import openai
import base64
import os

# Set OpenAI API key
openai.api_key = 'OPENA_API'

def encode_image(image_path):
    """
    Encodes an image to a base64 string.

    Args:
        image_path (str): The file path to the image to be encoded.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Image folder path
image_folder = "vehicle_snapshots1_2"

# Iterate through all files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        base64_image = encode_image(image_path)

        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use the appropriate vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please identify the vehicle in this image and output the information in 'vehicle model/ year range (A-B)/ vehicle type/ license plate' format. If you are unable to identify specific models, give a guess in 'vehicle model/ year range (A-B)/ vehicle type/ license plate' format.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # Use high resolution mode to improve recognition accuracy
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )

        # Print the vehicle information
        print(f"File: {filename}")
        print(response.choices[0].message.content)
        print()
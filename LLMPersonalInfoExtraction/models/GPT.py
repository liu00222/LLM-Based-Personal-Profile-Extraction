import base64
from openai import OpenAI

from .Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.api_key = api_keys[api_pos]
        self.set_API_key()
        
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.vision_model = OpenAI(api_key=self.api_key)
        
    def set_API_key(self):
        self.client = OpenAI(api_key=self.api_key)
        
    def query(self, msg, image_path=None):
        if image_path is None:
            completion = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "user", "content": msg}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )
            response = completion.choices[0].message.content

        else:
            base64_image = self.__encode_image(image_path)
            response = self.vision_model.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            response = response.choices[0].message.content
        return response

    def __encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
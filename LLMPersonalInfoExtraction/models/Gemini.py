import google.generativeai as palm
import google.ai.generativelanguage as gen_lang
import PIL.Image
import time

from .Model import Model
from ..utils import load_image


class Gemini(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos <= len(api_keys)), "Please enter a valid API key to use"
        self.api_key = api_keys[api_pos]
        self.set_API_key()
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        self.text_model = palm.GenerativeModel('gemini-pro')
        self.vision_model = palm.GenerativeModel('gemini-pro-vision')
        
    def set_API_key(self):
        palm.configure(api_key=self.api_key)

    def query(self, msg, image=None):
        trial = 0
        while trial < 5:
            try:
                return self.__do_query(msg, image)
            except:
                trial += 1
                if trial % 2 == 1:
                    time.sleep(0.5)
        return ''

    def __do_query(self, msg, image=None):
        if image is None:
            raw_response = self.text_model.generate_content(msg)

        elif type(image) == str:
            # We assume this is the path to the image
            img = load_image(image)
            if img is None:
                raise RuntimeError(f"Bad image: {image}")
            raw_response = self.vision_model.generate_content([msg, img], stream=True)
            raw_response.resolve()
        
        else:
            # print('here')
            raw_response = self.vision_model.generate_content([msg, image], stream=True)
            raw_response.resolve()
        
        return raw_response.text
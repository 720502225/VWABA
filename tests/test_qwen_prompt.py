
import unittest
from unittest.mock import MagicMock
from PIL import Image
from agent.prompts.prompt_constructor import MultimodalCoTPromptConstructor
from llms.lm_config import LMConfig

class TestQwenPromptConstructor(unittest.TestCase):
    def setUp(self):
        self.lm_config = MagicMock(spec=LMConfig)
        self.lm_config.provider = "qwen"
        self.lm_config.mode = "chat"
        self.lm_config.gen_config = {"max_obs_length": 100}
        
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.decode.return_value = "decoded"

        self.instruction_path = "agent/prompts/jsons/p_cot_id_act_tree_2s.json" # Dummy path, will mock json load if needed, but constructor loads it. 
        # Actually constructor loads it. I should mock json.load or provide a real file. 
        # Let's mock the constructor's init to avoid file loading issues for this unit test if possible, 
        # or just create a dummy json file.
        
    def test_get_lm_api_input(self):
        # Bypass __init__ to avoid file loading
        constructor = MultimodalCoTPromptConstructor.__new__(MultimodalCoTPromptConstructor)
        constructor.lm_config = self.lm_config
        constructor.tokenizer = self.tokenizer
        
        intro = "System Intro"
        examples = [("User Example", "Assistant Example", "d:/visualwebarena/agent/prompts/jsons/example_img.png")] # Need a dummy image path
        current = "Current Observation"
        
        # Create dummy images
        img = Image.new('RGB', (60, 30), color = 'red')
        # We need to mock Image.open to return our dummy image for the example
        
        with unittest.mock.patch('PIL.Image.open', return_value=img):
            message = constructor.get_lm_api_input(
                intro, 
                examples, 
                current, 
                img, 
                [img]
            )
            
        # Verify structure
        self.assertEqual(len(message), 4) # System, Example User, Example Assistant, Current User
        
        # System
        self.assertEqual(message[0]['role'], 'system')
        self.assertEqual(message[0]['content'][0]['text'], intro)
        
        # Example User
        self.assertEqual(message[1]['role'], 'user')
        self.assertIn("IMAGES", message[1]['content'][1]['text'])
        self.assertEqual(message[1]['content'][2]['type'], 'image_url')
        
        # Example Assistant
        self.assertEqual(message[2]['role'], 'assistant')
        self.assertEqual(message[2]['content'][0]['text'], "Assistant Example")
        
        # Current User
        self.assertEqual(message[3]['role'], 'user')
        self.assertIn("IMAGES", message[3]['content'][1]['text'])
        self.assertEqual(message[3]['content'][2]['type'], 'image_url')
        self.assertEqual(message[3]['content'][0]['text'], current)

if __name__ == '__main__':
    unittest.main()

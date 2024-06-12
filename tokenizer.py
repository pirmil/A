from typing import List, Dict
import re
import emoji
from nltk.tokenize import TweetTokenizer
import spacy


EMOJI_MAPPING = {
    '🤤': 'drooling',
    '😉': 'winking',
    '😏': 'smirking',
    '🤭': 'gossiping',
    '😚': 'kissing',
    '😘': 'kissing',
    '💖': 'love',
    '❤️': 'love',
    '🧡': 'love',
    '💙': 'love',
    '♥️': 'love',
    '💗': 'love',
    '🤍': 'love',
    '🤎': 'love',
    '💜': 'love',
    '💚': 'love',
    '🤟': 'love',
    '🥰': 'love',
    '😍': 'love',
    '🤔': 'thinking',
    '😌': 'relieved',
    '😱': 'scared',
    '😨': 'scared',
    '💢': 'angry',
    '😠': 'angry',
    '😤': 'angry',
    '😡': 'angry',
    '👿': 'angry',
    '🖕': 'angry',
    '😈': 'naughty',
    '🤗': 'hugging',
    '🤣': 'laughing',
    '😂': 'laughing',
    '😆': 'laughing',
    '🙂': 'laughing',
    '🤩': 'fascinated',
    '🤩': 'fascinated',
    '😶': 'speechless',
    '😮': 'shocked',
    '😲': 'shocked',
    '🤯': 'shocked',
    '🙄': 'disdain',
    '😴': 'sleepy',
    '🙄': 'annoyed',
    '😐': 'expressionless',
    '😑': 'expressionless',
    '🙃': 'sarcasm',
    '🧐': 'monocle',
    '🤨': 'perplexed',
    '💀': 'skull',
    '☠️': 'skull',
    '😬': 'grimace',
    '😷': 'mask',
    '😢': 'sad',
    '😭': 'sad',
    '😣': 'sad',
    '🥺': 'sad',
    '😔': 'sad',
    '😟': 'sad',
    '💔': 'disappointed',
    '😞': 'disappointed',
    '😩': 'upset',
    '☹️': 'frowning',
    '🤑': 'money',
    '💸': 'money',
    '💰': 'money',
    '💵': 'money',
    '💳': 'money',
    '🤓': 'nerd',
    '🤡': 'clown',
    '🐷': 'pig',
    '🐱': 'cat',
    '🐐': 'goat',
    '🙇': 'bowing',
    '🙇‍♀️': 'bowing',
    '🤷': 'shrugging',
    '🤷‍♂️': 'shrugging',
    '💁‍♀️': 'assistance',
    '🙅‍♀️': 'disapproval',
    '🙅‍♂️': 'disapproval',
    '🤦': 'facepalm',
    '🤦‍♀️': 'facepalm',
    '🤦‍♂️': 'facepalm',
    '🙋': 'question',
    '🙋': 'question',
    '😃': 'joy',
    '😊': 'joy',
    '😀': 'joy',
    '😄': 'joy',
    '☺️': 'joy',
    '🤖': 'robot',
    '👽': 'alien',
    '🛸': 'alien',
    '🚀': 'rocket',
    '🎵': 'music',
    '🎶': 'music',
    '🥳': 'party',
    '🎊': 'party',
    '🎂': 'party',
    '🎉': 'party',
    '✨': 'party',
    '😎': 'sunglasses',
    '😇': 'angel',
    '🌞': 'sun',
    '🙉': 'covering ears',
    '🙈': 'covering eyes',
    '🔒': 'padlock',
    '💍': 'ring',
    '💎': 'diamond',
    '📖': 'book',
    '⚖️': 'balance scale',
    '💣': 'bomb',
    '🔫': 'gun',
    '💥': 'explosion',
    '💡': 'light bulb',
    '💩': 'poop',
    '🗿': 'stoic',
    '👀': 'eyes',
    '💪': 'flexing',
    '🤲': 'praying',
    '👏': 'clapping',
    '👊': 'solidarity',
    '✊': 'solidarity',
    '✖️': 'disapproval',
    '❌': 'disapproval',
    '❎': 'disapproval',
    '💯': 'approval',
    '✅': 'approval',
    '☑️': 'approval',
    '✔️': 'approval',
    '👍': 'approval',
    '👌': 'approval',
    '✌️': 'peace',
    '🤞': 'hoping',
    '🧠': 'brain',
    '🏆': 'victory',
    '🏅': 'victory',
    '🥇': 'victory',
    '⚠️': 'warning',
    '🔥': 'fire',
    '🔮': 'crystal ball',
    '💹': 'increasing',
    '📈': 'increasing',
    '📉': 'decreasing',
    '❄️': 'snow',
    '💧': 'water',
    '💦': 'water',
    '🌊': 'water wave',
}

STOP_WORDS = ['a', 'about', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are', 'at',
    'b', 'be', 'because', 'been', 'before', 'being', 'but', 'by',
    'c', 'can', 
    'd', 'did', 'do', 'does', 'doing', 'don', 'down',
    'e', 'every', 'each',
    'f', 'for', 'from',
    'g',
    'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'him', 'his', 'how', "she'd", "he'd", "he'll", "she'll",
    'i', 'if', 'in', 'into', 'is', 'it', 'its', "it's", "it'll", "I'd", "I'll",
    'j', 'just',
    'k',
    'l', 'll',
    'm', 'more', 'most', 'my', 'me', 'myself',
    'n', 'now',
    'o', 'only', 'or', 'other', 'our', 'out', 'over',
    'p',
    'q',
    'r',
    's', 'same', 'she', 'should', 'so', 'some',
    't', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'to', 'too', "they'll", "they'd",
    'u', 'us',
    'v',
    'w', 'was', 'we', 'were', 'what', 'when', 'where', 'who', 'why', 'will', 'with', 'whom', 'whose', 'which',
    'x',
    'y', 'you', 'your', 'yourself', "you'll", "you'd",
    'z',
]

class CustomTokenizer:
    def __init__(self, stop_words: List[str], emoji_mapping: Dict[str, str], remove_urls=True, tokenizer='tweet_tokenizer'):
        self.emoji_mapping = emoji_mapping
        self.remove_urls = remove_urls
        self.tokenizer = tokenizer
        if tokenizer == 'tweet_tokenizer':
            self.tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True, match_phone_numbers=False)
        self.stop_words = stop_words

    def __call__(self, text: str):
        text = self.process_urls(text, self.remove_urls)
        text = self.process_emojis(text, self.emoji_mapping)
        tokens = self.tokenize(text, self.tokenizer)
        return tokens
    

    def tokenize(self, text: str, tokenizer: str):
        if tokenizer == 'tweet_tokenizer':
            tokens = self.tweet_tokenizer.tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words]
            return tokens

    def process_urls(self, text: str, remove=True, url_token='URL'):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        if remove:
            return re.sub(url_pattern, '', text)
        return re.sub(url_pattern, url_token, text)

    def normalize_whitespace(self, text: str):
        return re.sub(r'\s+', ' ', text)

    def process_emojis(self, text: str, emoji_mapping: dict):
        for emoj, value in emoji_mapping.items():
            text = text.replace(emoj, f' {value} ')
        # Remove any remaining emoji
        text = emoji.replace_emoji(text, '')
        text = self.normalize_whitespace(text)
        return text
    
custom_tokenizer = CustomTokenizer(STOP_WORDS, EMOJI_MAPPING)
text = "Check checked checks this out🍵🔥🤤❤️! 🤦Checking ⚖️🍵tim@gmail.com 🐍❤️ $11 http://example.com #Covid19 #Amazing #IAmHappy🍵🔥🤤❤️🍵 @elonmusk $Tesla.🏳️‍🌈 It's not what you'll  think @r@a! $TSLA she isn't happy"
print(text)
tokenized_text = ' '.join(custom_tokenizer(text))
tokenized_text

nlp = spacy.load('en_core_web_sm')
tokens = [token.lemma_ for token in nlp(tokenized_text) if token.lemma_ not in STOP_WORDS]
' '.join(tokens)

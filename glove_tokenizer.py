import re
from typing import Dict, List
import emoji

ABBREVIATION_MAPPING = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and "
}

class GloveTokenizer:
    def __init__(self, abbreviation_mapping: Dict[str, str], to_list=False):
        self.abbreviation_mapping = abbreviation_mapping
        self.to_list = to_list

    def __call__(self, text: str):
        text = self.replace_urls(text)
        text = self.remove_emails(text)
        text = self.replace_quotes(text)
        text = self.split_words_with_slashes(text)
        text = self.replace_users(text)
        text = self.remove_emojis(text)
        text = self.replace_numbers(text)
        text = self.replace_hashtags(text)
        text = self.expand_abbreviations(text.lower(), self.abbreviation_mapping)
        text = self.mark_repetition(text)
        text = self.mark_elongation(text)
        return self.convert_to_list(text, self.to_list)
    
    def replace_urls(self, text: str):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, "<URL>", text)
    
    def remove_emails(self, text: str):
        return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    
    def replace_quotes(self, text: str):
        text = re.sub(r"[‘’]", "'", text)
        text = re.sub(r"[“”]", '"', text)
        return text

    def split_words_with_slashes(self, text: str):
        return re.sub(r"/", " / ", text)

    def replace_users(self, text: str):
        return re.sub(r"@\w+", "<USER>", text)

    def remove_emojis(self, text: str):
        text = emoji.replace_emoji(text, '')
        return text
    
    def replace_numbers(self, text: str):
        return re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)

    def replace_hashtags(self, text: str):
        def hashtag_splitter(hashtag):
            hashtag_body = hashtag[1:]
            if hashtag_body.upper() == hashtag_body:
                return f"<HASHTAG> {hashtag_body} <ALLCAPS>"
            else:
                return "<HASHTAG> " + " ".join(re.split(r'(?=[A-Z])', hashtag_body))
        return re.sub(r"#\S+", lambda x: hashtag_splitter(x.group()), text)  
    
    def expand_abbreviations(self, text: str, abbreviation_mapping: Dict[str, str]):
        for abb, expand in abbreviation_mapping.items():
            text = text.replace(abb, expand)
        return text
    
    def mark_repetition(self, text: str):
        return re.sub(r"([!?.]){2,}", lambda x: f"{x.group(0)[0]} <REPEAT>", text)
     
    def mark_elongation(self, text: str):
        return re.sub(r"\b(\S*?)(.)\2{2,}\b", lambda x: f"{x.group(1)}{x.group(2)} <ELONG>", text)

    def normalize_whitespace(self, text: str):
        return re.sub(r'\s+', ' ', text)
    
    def convert_to_list(self, text: str, to_list: bool):
        if to_list:
            return self.normalize_whitespace(text).strip()
        else:
            return self.normalize_whitespace(text)
        

glove_tokenizer = GloveTokenizer(ABBREVIATION_MAPPING)
glove_tokenizer("I don't love love $$11.05 the https://nlp.stanford.edu new #FunnyRabbit #Tesla #lol mooodel @elonmusk!!🤤🤤 helloooo timothee@gmail.com")
# 'i do not love love $$<number> the <url> new <hashtag> funny rabbit <hashtag> tesla <hashtag> lol mooodel <user>! <REPEAT> hello <ELONG> '
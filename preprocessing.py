import re


def clean_text(x):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
              '*', '+', '\\', '•', '~', '@', '£',
              '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â',
              '█', '½', 'à', '…',
              '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
              '¥', '▓', '—', '‹', '─',
              '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
              '¾', 'Ã', '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
              '¹', '≤', '‡', '√', ]

    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')

    return x


def remove_names(x):
    for word in x.split():
        if word[0] == "@":
            x = x.replace(word, "")
    return x


def sep_digits(x):
    return " ".join(re.split('(\d+)', x))


def sep_punc(x):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~؛،؟؛.»«”'
    out = []
    for char in x:
        if char in punc:
            out.append(' '+char+' ')
        else:
            out.append(char)
    return ''.join(out)

damma = "ُ"
sukun = "ْ"
fatha = "َ"
kasra = "ِ"
shadda = "ّ"
tanweendam = "ٌ"
tanweenfath = "ً"
tanweenkasr = "ٍ"
tatweel = "ـ"

tashkil = (damma, sukun, fatha, kasra, shadda, tanweendam, tanweenfath, tanweenkasr, tatweel)


def remove_tashkil(word):
    w = [letter for letter in word if letter not in tashkil]
    return "".join(w)


def clean_arabic(x):
    return sep_punc(sep_digits(remove_tashkil(x)))


def normalize(some_string):
    normdict = {
        'ة': 'ه',
        'أ': 'ا',
        'إ': 'ا',
        'ي': 'ى',

    }
    out = [normdict.get(x, x) for x in some_string]
    return ''.join(out)



import re

teststring = "Die Situation der Länderparlamente, die kritisch erscheint, ist zugleich Herausforderung.\n Obgleich ich" \
             "kein Chinese bin, lasse ich mir doch berichten,\n daß es für die Begriffe „Krise” und „Chance“ nur ein\n" \
             "und dasselbe chinesische Schriftzeichen gebe.\n Das verspreche ich."


def cleanup(text: str) -> str:
    n = re.compile('[^.]\n')

    return re.sub(n, " ", text)

print(cleanup(teststring))
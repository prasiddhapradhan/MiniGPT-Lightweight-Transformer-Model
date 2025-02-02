import urllib.request

url = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"

# Download the file
response = urllib.request.urlopen(url)
text = response.read().decode("utf-8")

# Save the text file locally
with open("war_and_peace.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Download Complete: war_and_peace.txt saved!")
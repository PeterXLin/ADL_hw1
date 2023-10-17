import gdown


# download data
url = "https://drive.google.com/drive/folders/12F_1fETNF-xmeA0P2Vb6orDFQzdkihbL"
gdown.download_folder(url, quiet=True, use_cookies=False)

# download model
url = "https://drive.google.com/drive/folders/1__8diUSO5l4p_-7u6vMuvmcK4RRIhEbW"
gdown.download_folder(url, quiet=True, use_cookies=False)
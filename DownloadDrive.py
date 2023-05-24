import gdown

def descarregar():
    id = "1CCqE6FbnPmEjwVs6wWR4yqJ_WAVy-PKG"
    output = "./ResNext101_64x4d_PreTrained.pth"
    gdown.download(id=id, output=output, quiet=False)


    
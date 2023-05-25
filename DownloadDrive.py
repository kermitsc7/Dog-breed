import gdown

def descarregar():
    id = "17AWuv0xkSyhW3j2bIy6ndAG89YwVhczr"
    output = "./ResNext101_64x4d_PreTrained.pth"
    gdown.download(id=id, output=output, quiet=False)


    

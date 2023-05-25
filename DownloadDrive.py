import gdown

def descarregar():
    id = "17AWuv0xkSyhW3j2bIy6ndAG89YwVhczr"
    output = "./ResNext101_32x8d_PreTrained.pth"
    gdown.download(id=id, output=output, quiet=False)


    

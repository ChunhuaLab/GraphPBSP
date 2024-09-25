import os,sys


def mkdir(file):

    folder = os.path.exists(file)

    if not folder:                  
        os.makedirs(file)           
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print ("---  There is this folder!  ---")

 
    
    #return result

def main():
    path='../'+sys.argv[1]
    Type=sys.argv[2]
    
    dssppath=f'dssp_{Type}'
    mkdir(dssppath)
    
    for dir in os.listdir(path):
        
        name=dir[0:6]
        
        print(name+' computed....')
        #generate dssp files
        os.system('./dssp '+path+dir+f' > ./{dssppath}/'+name+'.dssp')
  
    os.system(f'mv {dssppath} ../example/features ')
  
if __name__=="__main__":
    main() 
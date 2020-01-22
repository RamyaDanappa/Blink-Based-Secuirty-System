import numpy as np 
import pandas as pd 
import pickle
from CalibTrnr import calibration
from CalibTrnr import encrypted
from detect_eyes import passrd
try:
    with open("users.pickle", "rb") as pic:
            users =  pickle.load(pic)
except:
    users = {}
try:
    with open("Encoder.pickle","rb") as epp:
        Encoder = pickle.load(epp) 
except:
    Encoder ={"1111":"A","1112":"B","1121":"C","1122":"D","1211":"E","1212":"F","1221":"G","1222":"H","2111":"I","2112":"J","2121":"K","2122":"L","2211":"M","2212":"N","2221":"O","2222":"P","111":"Q","112":"R","121":"S","122":"T","211":"U","212":"V","221":"W","222":"X","1":"Y","2":"Z"}


while True:
    print("\n\n\n\n\n\t\t\t\tWelcome to Secure login..\n\n\t\t\t\t.....................\n\n\t\t\t\t.....................\n\n\t\t\t\t.....................\n\n")
    print("\t\t\t\t Enter 1: Login , 2: CreateID, 3: Exit\n\n\t\t\t\t")
    choice = input()


    if choice == "2":
        print("\n\n\t\t\t\tEnter Your Username\n\n\t\t\t\t(Max len 32, alphabets, special charecters and numbers):\n\n\t\t\t\t")
        nusrnm = input()
        users[nusrnm] = []
        if nusrnm  in users.keys() and len(nusrnm) > 3:
            print("\n\n\t\t\t\tEntered Name:", nusrnm,"\n\n")
            
        else:
            print("\n\n\t\\t\t\tUsername already exsistsPlease start over!!\n\n\n\n\n\n")
            continue
        print("\n\n\t\t\t\tCalibration for password through blink is starting...\n\n\t\t\t\tSelect two different rates of blinking to setup password\n\n\t\t\t\t(Example: If type1 blink is a slow blink, the make sure type2 blink is faster.\n\n\t\t\t\t)")
        print("Starting Recording for Type 1 blink...\n\n")

        (x,y, z) = calibration(nusrnm, "t1")
        print(x," ",y,z)
        users[nusrnm].append(nusrnm + "_t1.pickle")
        print("\n\n\t\t\t\tStarting Recording for Type 2 blink...\n\n")
        (k,l,w) = calibration(nusrnm, "t2")
        print(k," ",l,w)
        users[nusrnm].append(nusrnm + "_t2.pickle")
        print("\n\n\t\t\t\tDone!!\n\n\t\t\t\tEnter password (combination of the previously set blink types)\n\n\t\t\t\tMax no.blinks is 64 and min is 8")
        opwd = passrd(nusrnm)
        epwd = encrypted(opwd)
        users[nusrnm].append(epwd)
        if len(users[nusrnm]) == 3:
            print("\n\n\t\t\t\tSuccessfully created ID:",nusrnm,"\n\t\t\t\tDEBUG pwd:",opwd, " length", len(opwd) )
        pickle.dump(users, open("users.pickle","wb"))

    elif choice == "3":
        break

    elif choice == "1":
        print("\n\n\t\t\t\tPleaseenter your UsedID:\n\n\t\t\t\t" )
        usnm = input()
        usnm = str(usnm)
        
        if usnm in users.keys():
            apwd = users[usnm][-1]
            opw = ""
            for c in apwd:
                for k,v in Encoder.items():
                    if v == c:
                        print("\n", v," ",c,"\n")
                        opw = opw + k
        
        cpwd = passrd(usnm)
        
        if opw and cpwd:
            print("\n\n\t\t\t\topwd:",opw, "\t\t entered:",cpwd)
            if cpwd==opw:
                print("\n\n\t\t\t\t\t\t\t LOGIN SUCCESS\n\n\n\n\n")
            else:
                print("\n\n\t\t\t\t Wrong password try again !!\n\n\n\n")
        else:
            print("\n\n\t\t\t\t\t Check your credentials!!")        

print("\n\n\n\n\t\t\t\t BYE!!")


#Importando as bibliotecas
import cv2
import imutils
import datetime

#funcao para saber qual vai ser a entrada (video salvo ou WebCam)
def opcao_entrada() : 
    
    while True:   
        opcao = input('''----- DETECTOR DE ARMAS ----- 
            
1 - Video salvo; 
2 - WebCam; 

Escolha a entrada que sera analisada: ''')  
    
        if opcao == '1': 
            camera = cv2.VideoCapture('/home/felipex/Documentos/UERR_CC/Inteligência Artificial_2020.1/source/gun-detect/data/gun4_2.mp4')
            return camera
            break
        elif opcao == '2': 
            camera = cv2.VideoCapture(0)  
            return camera
            break
        else: 
            print('\n-> Informe um numero valido.\n')
        
camera = opcao_entrada() 

#carregando o modelo para deteccao de arma
gun_cascade = cv2.CascadeClassifier('/home/felipex/Documentos/UERR_CC/Inteligência Artificial_2020.1/source/gun-detect/cascade.xml')

#inicializa o primeiro frame
firstFrame = None
gun_exist = False

#fazendo um laco para analizar cada frame
while True:
    (grabbed, frame) = camera.read()

    # interrompe o laco se o frame nao puder ser capturado
    if not grabbed:
        break

    # redimensionar o frame, convertê-lo em tons de cinza e desfocá-lo
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # fazendo a deteccao para cada frame
    gun = gun_cascade.detectMultiScale(gray, 1.2, 7, minSize = (60, 60))
    
    
    if len(gun) > 0:
        gun_exist = True
        
    for (x,y,w,h) in gun:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]    

    # se o firstframe = None , inicializa-lo
    if firstFrame is None:
        firstFrame = gray
        continue

    # data e hora no frame
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # mostrando o video
    cv2.imshow("Video", frame)
    
    # para sair pressiona a tecla 'q'
    if cv2.waitKey(1) == 27:
        break

if gun_exist:
    print("\nARMA(s) DETECTADA(s).")
else:
    print("\nNENHUMA ARMA DETECTADA.")

#fechando a camera e a janela
camera.release()
cv2.destroyAllWindows()







from aart_func import *
from params import * 
import subprocess
import itertools

# Params
spin_case = (0.94)
iCase = (17, 45, 75)
dxCase = (0.02, 0.04, 0.08)  # dx0=dx1=dx2, así que es solo un conjunto
Noise = (0.2,0.4)

# params map
combinations = list(itertools.product(SpinCase, iCase, dxCase, Noise))
N_com = len(combinations)

###### When you want to do all the light curve ######
#for i in range(N_com):
#    i_fname = r"inoisy_n%s_i%s_ft%s_snap%s.h5"%({noise},{spin_case},f_tM,snapshots)


#    spin_case = combinations[i][0]
#    i_case = combinations[i][1]
#    dx0 = combinations[i][2]

#    print("Working with the parameters a = %s , theta = %s , dx = %s, noise = %s "%(spin_case,i_case,dx0,noise))
#    subprocess.run(["python", "LightCurve.py", str(spin_case), str(i_case), str(dx0),str(i_fname)])

for i in range(N_com):
    spin_case = combinations[i][0]
    i_case = combinations[i][1]
    dx0 = combinations[i][2]  
    noise = combinations[i][3]  



    print("making the script with the parameters a = %s , theta = %s , dx = %s, noise = %s "%(spin_case,i_case,dx0,noise))
    contenido = f"""
    
    i_fname = r"inoisy_n%s_i%s_ft%s_snap%s.h5"%({noise},{spin_case},f_tM,snapshots)
    
    spin_case = {spin_case}
    i_case = {i_case}
    dx0 = {dx0}
    
    print("Working with the parameters a = %s , theta = %s , dx = %s, noise = %s "%(spin_case, i_case, dx0, noise))
    subprocess.run(["python", "LightCurve.py", str(spin_case), str(i_case), str(dx0), str(i_fname)])
    """
    # Nombre del archivo que quieres crear
    nombre_archivo = r"LightCurve_Generation_a%s_i%s_dx%s_noi%s.py"%(spin_case,i_case,dx0,noise)


    # Crear y escribir el archivo
    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(contenido)
    
    print(f"Archivo '{nombre_archivo}' creado exitosamente.")

    
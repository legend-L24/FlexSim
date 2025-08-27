from ase.io import read, write
import ase.build
import os

ele_num = {
    "H": 1,
    "Li": 3,
    "Be": 4,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 9,
    "Mg": 10,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8,
    "K": 9,
    "Ca": 10,
    "Sc": 11,
    "Ti": 12,
    "V": 13,
    "Cr": 14,
    "Mn": 15, 
    "Fe": 16,
    "Co": 17,
    "Ni": 18,
    "Cu": 11,
    "Zn": 12,
    "Ga": 13,
    "Ge": 4,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Kr": 8,
    "Rb": 9,
    "Sr": 10,
    "Y": 11,
    "Zr": 12,
    "Nb": 13,
    "Mo": 14,
    "Tc": 15,
    "Ru": 16,
    "Rh": 17,
    "Pd": 18,
    "Ag": 11,
    "Cd": 12,
    "In": 13,
    "Sn": 4,
    "Sb": 5,
    "Te": 6,
    "I": 7,
    "Xe": 8,
    "Cs": 9,
    "Ba": 10,
    "La": 11,
    "Hf": 12,
    "Ta": 13,
    "W": 14,
    "Re": 15,
    "Os": 16,
    "Ir": 17,
    "Pt": 18,
    "Au": 11,
    "Hg": 12,
    "Tl": 13,
    "Bi": 5,
}

def ABC(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    new_lines = []
    inside_cell_block = False

    for line in lines:
        if line.strip() == '&CELL':
            inside_cell_block = True
            new_lines.append(line)
            new_lines.append('A '+ str(cell[0,0])+' '+str(cell[0,1])+' '+str(cell[0,2]) + '\n')
            new_lines.append('B '+ str(cell[1,0])+' '+str(cell[1,1])+' '+str(cell[1,2]) + '\n')
            new_lines.append('C '+ str(cell[2,0])+' '+str(cell[2,1])+' '+str(cell[2,2]) + '\n')
        elif line.strip() == '&END CELL':
            inside_cell_block = False
            new_lines.append(line)
        elif not inside_cell_block:
            new_lines.append(line)
    with open(filename, 'w') as file:
        file.writelines(new_lines)

def ELEM(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    new_lines = []
    inside_cell_block = False

    for line in lines:
        if '&END COORD' in line:
            inside_cell_block = True
            new_lines.append(line)
            for elements in element:
                new_lines.append('&KIND '+ str(elements) + '\n')
                new_lines.append('ELEMENT '+ str(elements) + '\n')
                new_lines.append('BASIS_SET DZVP-MOLOPT-SR-GTH-q'+str(ele_num[elements]) + '\n')
                new_lines.append('POTENTIAL GTH-PBE-q'+str(ele_num[elements]) + '\n')
                new_lines.append('&END KIND'+'\n')
        elif '&END SUBSYS' in line:
            inside_cell_block = False
            new_lines.append(line)
        elif not inside_cell_block:
            new_lines.append(line)
    with open(filename, 'w') as file:
        file.writelines(new_lines)



ciffile = './initial.cif'
file = read(ciffile, format='cif')
cell = file.get_cell()
element = set(file.get_chemical_symbols())

write('coord.xyz', file, format='xyz')
command = f"sed -i '1,2d' coord.xyz"
os.system(command)
ABC('input.inp')
ELEM('input.inp')

import numpy as np

bit = 16
index_len = int(256 / bit)

triangular_index = np.zeros([index_len, index_len, index_len, index_len, 4+3])

# x y
# z t
for x in range(bit):
    for y in range(bit):
        for z in range(bit):
            for t in range(bit):
                # 24 case
                #xyzt
                sorted_array = [x,y,z,t]
                sorted_array.sort(reverse=True)
                for i in range(4):
                    triangular_index[x][y][z][t][i] = sorted_array[i]
                if x > y and y > z and z > t:
                    triangular_index[x][y][z][t][4] = int('0b1000', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1100', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1110', base=2)
                #xytz
                elif x > y and y > t and t > z:
                    triangular_index[x][y][z][t][4] = int('0b1000', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1100', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1101', base=2)
                #xtyz
                elif x > t and t > y and y > z:
                    triangular_index[x][y][z][t][4] = int('0b1000', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1001', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1101', base=2)
                #txyz
                elif t > x and x > y and y > z:
                    triangular_index[x][y][z][t][4] = int('0b0001', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1001', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1101', base=2)
                #xzyt
                elif x > z and z > y and y > t:
                    triangular_index[x][y][z][t][4] = int('0b1000', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1010', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1110', base=2)
                #xzty
                elif x > z and z > t and t > y:
                    triangular_index[x][y][z][t][4] = int('0b1000', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1010', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1011', base=2)
                #xtzy
                elif x > t and t > z and z > y:
                    triangular_index[x][y][z][t][4] = int('0b1000', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1001', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1011', base=2)
                #txzy
                elif t > x and x > z and z > y:
                    triangular_index[x][y][z][t][4] = int('0b0001', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1001', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1011', base=2)
                #zxyt
                elif z > x and x > y and y > t:
                    triangular_index[x][y][z][t][4] = int('0b0010', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1010', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1110', base=2)
                #zxty
                elif z > x and x > t and t > y:
                    triangular_index[x][y][z][t][4] = int('0b0010', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1010', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1011', base=2)
                #ztxy
                elif z > t and t > x and x > y:
                    triangular_index[x][y][z][t][4] = int('0b0010', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0011', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1011', base=2)
                #tzxy
                elif t > z and z > x and x > y:
                    triangular_index[x][y][z][t][4] = int('0b0001', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0011', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1011', base=2)
                #yxzt
                elif y > x and x > z and z > t:
                    triangular_index[x][y][z][t][4] = int('0b0100', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1100', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1110', base=2)
                #yxtz
                elif y > x and x > t and t > z:
                    triangular_index[x][y][z][t][4] = int('0b0100', base=2)
                    triangular_index[x][y][z][t][5] = int('0b1100', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1101', base=2)
                #ytxz
                elif y > t and t > x and x > z:
                    triangular_index[x][y][z][t][4] = int('0b0100', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0101', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1101', base=2)
                #tyxz
                elif t > y and y > x and x > z:
                    triangular_index[x][y][z][t][4] = int('0b0001', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0101', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1101', base=2)
                #yzxt
                elif y > z and z > x and x > t:
                    triangular_index[x][y][z][t][4] = int('0b0100', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0110', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1110', base=2)
                #yztx
                elif y > z and z > t and t > x:
                    triangular_index[x][y][z][t][4] = int('0b0100', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0110', base=2)
                    triangular_index[x][y][z][t][6] = int('0b0111', base=2)
                #ytzx
                elif y > t and t > z and z > x:
                    triangular_index[x][y][z][t][4] = int('0b0100', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0101', base=2)
                    triangular_index[x][y][z][t][6] = int('0b0111', base=2)
                #tyzx
                elif t > y and y > z and z > x:
                    triangular_index[x][y][z][t][4] = int('0b0001', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0101', base=2)
                    triangular_index[x][y][z][t][6] = int('0b0111', base=2)
                #zyxt
                elif z > y and y > x and x > t:
                    triangular_index[x][y][z][t][4] = int('0b0010', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0110', base=2)
                    triangular_index[x][y][z][t][6] = int('0b1110', base=2)
                #zytx
                elif z > y and y > t and t > x:
                    triangular_index[x][y][z][t][4] = int('0b0010', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0110', base=2)
                    triangular_index[x][y][z][t][6] = int('0b0111', base=2)
                #ztyx
                elif z > t and t > y and y > x:
                    triangular_index[x][y][z][t][4] = int('0b0010', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0011', base=2)
                    triangular_index[x][y][z][t][6] = int('0b0111', base=2)
                #tzyx
                else:
                    triangular_index[x][y][z][t][0] = t
                    triangular_index[x][y][z][t][1] = z
                    triangular_index[x][y][z][t][2] = y
                    triangular_index[x][y][z][t][3] = x
                    triangular_index[x][y][z][t][4] = int('0b0001', base=2)
                    triangular_index[x][y][z][t][5] = int('0b0011', base=2)
                    triangular_index[x][y][z][t][6] = int('0b0111', base=2)
np.save("/home/v-gyin/github/VSR_LUT_hidden/triangular_index.npy", triangular_index)
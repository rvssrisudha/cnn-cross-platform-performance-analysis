#################### Weight & Bias in HEX of Convolution Layer1 ####################

# Calibration
int_conv1_weight_1 = torch.tensor((model.conv1.weight.data[0][0] * 128), dtype = torch.int32)
int_conv1_weight_2 = torch.tensor((model.conv1.weight.data[1][0] * 128), dtype = torch.int32)
int_conv1_weight_3 = torch.tensor((model.conv1.weight.data[2][0] * 128), dtype = torch.int32)
int_conv1_bias = torch.tensor((model.conv1.bias.data * 128), dtype = torch.int32)

print("Signed")
print(int_conv1_weight_1)
print(int_conv1_weight_2)
print(int_conv1_weight_3)
print(int_conv1_bias)

# 2's Complement
for i in range(5):
    for j in range(5):
        if int_conv1_weight_1[i][j] < 0:
            int_conv1_weight_1[i][j] += 256
        if int_conv1_weight_2[i][j] < 0:
            int_conv1_weight_2[i][j] += 256
        if int_conv1_weight_3[i][j] < 0:
            int_conv1_weight_3[i][j] += 256

for k in range(3):
    if int_conv1_bias[k] < 0:
        int_conv1_bias[k] += 256

print ("Unsigned")
print(int_conv1_weight_1)
print(int_conv1_weight_2)
print(int_conv1_weight_3)
print(int_conv1_bias)

np.savetxt('conv1_weight_1.mem', int_conv1_weight_1, fmt='%1.2x',delimiter = " ")
np.savetxt('conv1_weight_2.mem', int_conv1_weight_2, fmt='%1.2x',delimiter = " ")
np.savetxt('conv1_weight_3.mem', int_conv1_weight_3, fmt='%1.2x',delimiter = " ")
np.savetxt('conv1_bias.mem', int_conv1_bias, fmt='%1.2x',delimiter = " ")


#################### Weight & Bias in HEX of Convolution Layer2 ####################

# Calibration
# print(np.shape(model.conv2.weight))
int_conv2_weight_11 = torch.tensor((model.conv2.weight.data[0][0]* 128), dtype = torch.int32)
int_conv2_weight_12 = torch.tensor((model.conv2.weight.data[0][1]* 128), dtype = torch.int32)
int_conv2_weight_13 = torch.tensor((model.conv2.weight.data[0][2]* 128), dtype = torch.int32)

int_conv2_weight_21 = torch.tensor((model.conv2.weight.data[1][0] * 128), dtype = torch.int32)
int_conv2_weight_22 = torch.tensor((model.conv2.weight.data[1][1] * 128), dtype = torch.int32)
int_conv2_weight_23 = torch.tensor((model.conv2.weight.data[1][2] * 128), dtype = torch.int32)

int_conv2_weight_31 = torch.tensor((model.conv2.weight.data[2][0] * 128), dtype = torch.int32)
int_conv2_weight_32 = torch.tensor((model.conv2.weight.data[2][1] * 128), dtype = torch.int32)
int_conv2_weight_33 = torch.tensor((model.conv2.weight.data[2][2] * 128), dtype = torch.int32)

int_conv2_bias = torch.tensor((model.conv2.bias.data * 128), dtype = torch.int32)

print ("Signed")
print(int_conv2_weight_11)
print(int_conv2_weight_12)
print(int_conv2_weight_13, '\n')

print(int_conv2_weight_21)
print(int_conv2_weight_22)
print(int_conv2_weight_23, '\n')

print(int_conv2_weight_31)
print(int_conv2_weight_32)
print(int_conv2_weight_33, '\n')

print(int_conv2_bias)

# 2's Complement
for i in range(5):
    for j in range(5):
        if int_conv2_weight_11[i][j] < 0:
            int_conv2_weight_11[i][j] += 256
        if int_conv2_weight_12[i][j] < 0:
            int_conv2_weight_12[i][j] += 256
        if int_conv2_weight_13[i][j] < 0:
            int_conv2_weight_13[i][j] += 256

        if int_conv2_weight_21[i][j] < 0:
            int_conv2_weight_21[i][j] += 256
        if int_conv2_weight_22[i][j] < 0:
            int_conv2_weight_22[i][j] += 256
        if int_conv2_weight_23[i][j] < 0:
            int_conv2_weight_23[i][j] += 256

        if int_conv2_weight_31[i][j] < 0:
            int_conv2_weight_31[i][j] += 256
        if int_conv2_weight_32[i][j] < 0:
            int_conv2_weight_32[i][j] += 256
        if int_conv2_weight_33[i][j] < 0:
            int_conv2_weight_33[i][j] += 256

for k in range(3):
    if int_conv2_bias[k] < 0:
        int_conv2_bias[k] += 256

print ("Unsigned")
print(int_conv2_weight_11)
print(int_conv2_weight_12)
print(int_conv2_weight_13, '\n')

print(int_conv2_weight_21)
print(int_conv2_weight_22)
print(int_conv2_weight_23, '\n')

print(int_conv2_weight_31)
print(int_conv2_weight_32)
print(int_conv2_weight_33, '\n')

print(int_conv2_bias)

np.savetxt('conv2_weight_11.mem', int_conv2_weight_11, fmt='%1.2x',delimiter = " ")
np.savetxt('conv2_weight_12.mem', int_conv2_weight_12, fmt='%1.2x',delimiter = " ")
np.savetxt('conv2_weight_13.mem', int_conv2_weight_13, fmt='%1.2x',delimiter = " ")

np.savetxt('conv2_weight_21.mem', int_conv2_weight_21, fmt='%1.2x',delimiter = " ")
np.savetxt('conv2_weight_22.mem', int_conv2_weight_22, fmt='%1.2x',delimiter = " ")
np.savetxt('conv2_weight_23.mem', int_conv2_weight_23, fmt='%1.2x',delimiter = " ")

np.savetxt('conv2_weight_31.mem', int_conv2_weight_31, fmt='%1.2x',delimiter = " ")
np.savetxt('conv2_weight_32.mem', int_conv2_weight_32, fmt='%1.2x',delimiter = " ")
np.savetxt('conv2_weight_33.mem', int_conv2_weight_33, fmt='%1.2x',delimiter = " ")

np.savetxt('conv2_bias.mem', int_conv2_bias, fmt='%1.2x',delimiter = " ")

#################### Weight & Bias in HEX of Fully Connected Layer ####################

print(np.shape(model.fc_1.weight))
print((model.fc_1.weight * 128).int())

print(np.shape(model.fc_1.bias))
print((model.fc_1.bias * 128).int())

int_fc_weight = (model.fc_1.weight * 128).int()
int_fc_bias = (model.fc_1.bias * 128).int()

# 2's Complement
for i in range(10):
    for j in range(48):
        if int_fc_weight[i][j] < 0 :
            int_fc_weight[i][j] += 256
    if int_fc_bias[i] < 0 :
        int_fc_bias[i] += 256

print(int_fc_weight)
print(int_fc_bias)

np.savetxt('fc_weight.mem', int_fc_weight, fmt='%1.2x',delimiter = " ")
np.savetxt('fc_bias.mem', int_fc_bias, fmt='%1.2x',delimiter = " ")

import time

start_time = time.time()
test()
end_time = time.time()

execution_time = end_time - start_time
print(f"Test function execution time: {execution_time:.4f} seconds")

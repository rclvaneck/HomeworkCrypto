# Renee van Eck
# 10368140
# program to find the plaintext of a vigenere ciphertext using frequency analysis

from random import randint
import numpy as np

# Letter frequency derived from the enron dataset (1.4GB of emails)
BYTE_FREQ = np.array([0, 7.740026656201475e-09, 0, 0, 3.518193934637034e-09, 0, 0, 0, 0, 0.002897324881854685, 0.02344302299277087, 0, 0, 0.009862113282726168, 0, 4.221832721564441e-09, 7.036387869274069e-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1295492407745933, 0.000305607212493459, 0.001756099466086206, 0.000108143652440447, 0.0003966102240843544, 0.0001181486923517678, 0.0003833811112513322, 0.001619407780782541, 0.001433126448331379, 0.001482979256385186, 0.001935847512400747, 0.0002398282441363373, 0.01150195332660351, 0.01623299677276915, 0.01715644316942845, 0.008147220311280005, 0.01159028251080408, 0.005841498737781784, 0.005715933692615801, 0.002657272176945318, 0.002255091948223646, 0.002751986179498468, 0.001907568973192922, 0.002714807313274798, 0.002325888564770347, 0.002128292016986606, 0.009627495483807029, 0.0009909429472953102, 0.002558781745022728, 0.007031162647642346, 0.003816421383533199, 0.001074359325485554, 0.006216030887648717, 0.008013247486249027, 0.002414553384672283, 0.008119933902761747, 0.004179012783185974, 0.009052412206890046, 0.003757220734195061, 0.001835971615706697, 0.002071553399763928, 0.005287628763013088, 0.002384282844058666, 0.001286552156265318, 0.002285764970223386, 0.005273746673385798, 0.008862666157052054, 0.005802914001261833, 0.004189429451787647, 0.0003446619797230779, 0.004893452425492716, 0.006553111159442652, 0.00773473107069106, 0.002015061056116674, 0.00114695022093892, 0.001627605876289032, 0.002883371021071128, 0.0006554901920155382, 0.0002403411968120074, 0.0001503239831615973, 0.001137383547991855, 0.0001508488976966452, 0, 0.002794663279203189, 3.733929586708977e-05, 0.04851478331300015, 0.008761197925783186, 0.02546572486247478, 0.01897490473392245, 0.07340723254660156, 0.009828748842366432, 0.01174477696105579, 0.01990229361870491, 0.0403587914406023, 0.001962360621892172, 0.005704511524187609, 0.02601219044628864, 0.02019394274858209, 0.05100791555920255, 0.05153407975673597, 0.01126567142195328, 0.0005943826815648275, 0.04330814548527876, 0.035208213922313, 0.04534239547475373, 0.01518359340413955, 0.006528112280620695, 0.008003267777334035, 0.001689369178089159, 0.01137677950460306, 0.0009253103358058694, 4.147950648937063e-06, 0.0001045888692888898, 4.302751182061093e-06, 0.000132569769289845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.221832721564441e-09, 0, 0, 0, 0, 0, 0, 0, 0, 1.477641452547554e-08, 0, 0, 0, 0, 3.870013328100738e-08, 5.629110295419255e-09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.814555147709627e-09, 0, 0, 0, 0, 5.629110295419255e-09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.629110295419255e-09, 0, 0, 0, 0, 0, 7.036387869274069e-10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.147304230056289e-09, 0, 0, 0, 0, 0, 0, 0, 0, 4.221832721564441e-09, 0, 0, 3.518193934637034e-09, 0, 2.814555147709627e-09, 7.036387869274069e-10, 0, 1.125822059083851e-08, 0, 0, 0, 8.443665443128881e-09, 0, 0, 0, 1.407277573854814e-09, 0, 0, 2.814555147709627e-09, 0, 3.518193934637034e-09, 0, 4.221832721564441e-09, 0, 0, 0, 3.518193934637034e-09, 0, 0, 7.036387869274068e-09],dtype=float)

# Takes a hex character list and converts this to ascii ints list
def char_to_ascii(char_list):
    c_hex = [char_list[i:i+2] for i in range(0, len(char_list), 2)]
    c_hex.remove('\n') # get rid of \n
    c_ascii = [int(h, 16) for h in c_hex]
    return c_ascii

# creates the ascii frequency table of a certain text
# input table with ascii ints
# output, numpy array of len(256) with the frequency of each number
# corresponding to an ascii letter
def frequency_table(text):
    freq = []
    for i in range(256):
        freq.append(text.count(i))
    freq_t = np.array(freq) / len(text)
    return np.array(freq) / len(text)

# in order to derive the key length we need to find the max std of the stream
def derive_key_length(c_text):
    key_length = 0
    max_dev = 0
    for t in range(1, 14):
        q = np.zeros(256)
        pos = 0
        while pos < len(c_text):
            q[c_text[pos]] += 1
            pos += t
        q /= q.sum() #normalize stream
        q = q ** 2 #calculate std dev
        cur_dev = q.sum()
        if cur_dev > max_dev:
            max_dev = cur_dev
            key_length = t
    return key_length


# to derive the key we need to find
def derive_key(c_text, k_length):
    key = np.zeros(k_length)
    for i in range(k_length):
        key_byte_max = 0
        for j in range(256):
            stream = np.zeros(256)
            pos = i
            while pos < len(c_text):
                stream[c_text[pos] ^ j] += 1 #XOR the ciphertext char with j
                pos += k_length
            stream /= stream.sum() # normalize stream
            current_byte_stream = (stream * BYTE_FREQ).sum()
            if current_byte_stream > key_byte_max:
                key_byte_max = current_byte_stream
                key[i] = j
    return key


def main():
    cipher_text = open('ciphertext.txt', 'r')
    c = char_to_ascii(cipher_text.read())
    freq_cipher = frequency_table(c)
    key_l = derive_key_length(c)
    key = derive_key(c, key_l).astype(int)
    plaintext = open('plaintext.txt', 'w')
    # decrypt the ciphertext into plaintext
    for i in range(len(c)):
        char = chr(c[i] ^ key[i % key_l])
        plaintext.write(char)

if __name__ == "__main__":
    main()

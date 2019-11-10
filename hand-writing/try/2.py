import struct
from PIL import Image
import os
count = 0
path = 'I:/sx2/'
for z in range(1241,1301):
    f = open('D:/shouxie/' + str(z) + '-c.gnt','rb')
    while True:
        count += 1
        tmp = f.read(4)
        if(len(tmp)==0):
            break
        length_bytes = struct.unpack('<I', tmp)[0]
        tag_code = f.read(2).decode('gb2312')
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        if(count%500==0):
            print('z='+str(z)+' count='+str(count))
        
        im = Image.new('RGB',(width,height))
        img_array = im.load()
        for x in range(0,height):
            for y in range(0,width):
                pixel = struct.unpack('<B',f.read(1))[0]
                img_array[y,x]=(pixel,pixel,pixel)
        filename = str(count) + '.png'
        if(os.path.exists(path + tag_code)):
            filename = path + tag_code + '/' + filename
            im.save(filename)
        else:
            os.makedirs(path + tag_code)
            filename = path + tag_code + '/' + filename
            im.save(filename)
    f.close()

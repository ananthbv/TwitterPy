            im = Images1['A']
            pixels = list(im.getdata())
            width, height = im.size
            pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
            print len(pixels)
            print pixels[60][56]
            for x in range(len(pixels)):
                print pixels[x]




# taken from - http://www.labbookpages.co.uk/software/imgProc/blobDetection.html
def checkObjectCountType(problem):
    solution = -1
    imgData = Images1['A'].getdata()
    pixels = Images1['A'].load()
    print 'pixels =', pixels
    maxxy = Images1['A'].size
    objects = {}
    nearby = {}
    objectNum = 0
    for x in range(maxxy[0]):
        for y in range(maxxy[1]):
            #print 'pixel =', pixels[x, y]
            #print x, y, pixels[x, y]
            if pixels[x, y] == 0:
                obj = checkIfInsideObject(x, y, nearby)
                if obj == -1:
                    obj = objectNum
                    print 'objectNum =', objectNum, x, y
                    objects[obj] = []
                    objectNum += 1
                objects[obj].append([x, y])
                nearby[obj] = getNearbyPixels(x, y, pixels)

    print 'Number of objects = ', objects
    return solution

def checkIfInsideObject(x, y, nearby):
    obj = -1
    for obj in nearby:
        if [x, y] in nearby[obj]:
            return obj
    return obj

def getNearbyPixels(x, y, pixels):
    nearby = []
    for x1, y1 in [[x-1, y], [x-1, y-1],
                   [x, y-1], [x, y+1],
                   [x+1,y+1], [x+1, y],
                   [x-1, y+1], [x+1, y-1]]:
        if pixels[x1, y1] == 0:
            nearby.append([x1, y1])
    #print 'nearby = ', nearby
    return nearby


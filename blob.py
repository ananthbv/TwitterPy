# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageMath, ImageOps
import math, operator
import copy

Images = {}
rgbImages = {}
grayscaleImages = {}
Images1 = {}

SIMILARITY_THRESHOLD = 550
solutionImageList = []

def checkObjectsCountType(objCounts, pixelCounts):
    solution = -1
    #return solution
    objCountsABC = [objCounts['A'], objCounts['B'], objCounts['C']]
    pixelsABC = [pixelCounts['A'] * 1.0 / objCounts['A'],
                 pixelCounts['B'] * 1.0 / objCounts['B'],
                 pixelCounts['C'] * 1.0 / objCounts['C']]
    objCountsDEF = [objCounts['D'], objCounts['E'], objCounts['F']]
    pixelsDEF = [pixelCounts['D'] * 1.0 / objCounts['D'],
                 pixelCounts['E'] * 1.0 / objCounts['E'],
                 pixelCounts['F'] * 1.0 / objCounts['F']]

    pixelsABCSorted = sorted(pixelsABC)
    pixelsDEFSorted = sorted(pixelsDEF)
    if sorted(objCountsABC) != sorted(objCountsDEF):
        return -1
    for i in range(3):
        if pixelsABCSorted[i] < pixelsDEFSorted[i]:
            if pixelsABCSorted[i] / pixelsDEFSorted[i] < 0.95:
                return -1
        else:
            if pixelsDEFSorted[i] / pixelsABCSorted[i] < 0.95:
                return -1


    print 'type is set of similar objects + count'
    #print 'objCountsABC, pixelsABC =', objCountsABC, pixelsDEF
    #print 'objCountsDEF, pixelsDEF =', objCountsDEF, pixelsABC
    for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
        objCountsGHI = [objCounts['G'], objCounts['H'], objCounts[i]]
        #print 'checking', i
        pixelsGHI = [pixelCounts['G'] * 1.0 / objCounts['G'],
                     pixelCounts['H'] * 1.0 / objCounts['H'],
                     pixelCounts[i] * 1.0 / objCounts[i]]
        #print 'objCountsGHI, pixelsGHI =', objCountsGHI, pixelsGHI
        #print 'sorted objCountsGHI, objCountsABC =', sorted(objCountsGHI), sorted(objCountsABC)
        if sorted(objCountsABC) != sorted(objCountsGHI):
            #print 'solution is not', i
            continue

        pixelsGHISorted = sorted(pixelsGHI)
        same = True
        for x in range(3):
            if pixelsABCSorted[x] < pixelsGHISorted[x]:
                if pixelsABCSorted[x] / pixelsGHISorted[x] < 0.95:
                    same = False
                    break
            else:
                if pixelsGHISorted[x] / pixelsABCSorted[x] < 0.95:
                    same = False
                    break

        if same:
            solution = int(i)

    if solution != -1:
        print 'returning', solution
        return solution
        #print 'objCountsGHI, pixelsGHI =', objCountsGHI, pixelsGHI
    return solution

# taken from - http://www.labbookpages.co.uk/software/imgProc/blobDetection.html
def getObjectsPixelData(im):
    #print 'getObjectsPixelData called'
    #im = Images1['B']
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    #print len(pixels)
    objects = {}
    labels = {}
    nearby = {}
    objectNum = 0
    for x in range(height):
        for y in range(width):
            if pixels[x][y] == 0:
                #print 'pixel ', x, y, 'is 0'
                obj = checkIfInsideObject(x, y, nearby, objects, pixels)
                #obj = checkIfInsideObject(x, y, objects)
                if obj == -1:
                    #print 'pixel is not near known objects, creating new object ', objectNum
                    obj = objectNum
                    #print 'objectNum =', objectNum, x, y
                    objects[obj] = []
                    objectNum += 1
                objects[obj].append([x, y])
                labels[str(x) + ',' + str(y)] = obj
                nearby[obj] = getNearbyPixels(x, y, pixels)

    
    #print 'labels before len, values =', len(labels.keys()), labels.values()
    
    combines = {}
    for x in range(height):
        for y in range(width):
            if pixels[x][y] != 0:
                continue
            nearby = getNearbyPixels(x, y, pixels)
            curr = str(x) + ',' + str(y)
            #print 'curr, labels[curr] =', curr, labels[curr]
            for p in nearby:
                nearbyPixel = str(p[0]) + ',' + str(p[1])
                if labels[curr] != labels[nearbyPixel]:
                    if labels[curr] < labels[nearbyPixel]:
                        #print 'labels[curr], labels[nearbyPixel] =', labels[curr], labels[nearbyPixel]
                        combines[labels[curr]] = labels[nearbyPixel]
                    else:
                        combines[labels[nearbyPixel]] = labels[curr]
                    
    if combines.keys():
        #print 'combines =', combines
        for k, v in combines.iteritems():
            for pix in objects[v]:
                objects[k].append(pix)
            objects.pop(v, None)
	return objects
	
	
	
	# OLD LOGIC - TAKES A LOOOOOONG TIME
    #print 'labels after len, values =', len(labels.keys()), labels.values()
    '''for i in range(1, len(labels.keys())):
        [xcurr, ycurr] = labels[i]
        [xprev, yprev] = labels[i - 1]
        if abs(xcurr - xprev) == 1 or abs(ycurr - yprev) == 1:
            if labels[i] != labels[i - 1]:
                print 'xcurr, ycurr, labelcurr, xprev, yprev, labelprev =', xcurr, ycurr, labels[i], xprev, yprev, labels[i - 1]
    
    objects = {}
    for v in labels.values():
        objects[v] = []
    #print 'objects =', objects
    for k, v in labels.iteritems():
        objects[v].append(k)
    #print 'objects =', objects    
    #print 'Number of objects = ', objects.keys()
    #for obj in objects.keys():
    #    #print 'len of obj', obj, len(objects[obj])
    #    print obj
    #    print sorted(objects[obj])
    #    print ''
    return objects
    combines = {}
    objectsCopy = copy.deepcopy(objects)
    for obj, objPixels in objectsCopy.iteritems():
        for pixel in reversed(objPixels):
            nearby = getNearbyPixels(pixel[0], pixel[1], pixels)
            for obj1, obj1Pixels in objects.iteritems():
                for pixel1 in sorted(obj1Pixels):
                    if [pixel1[0], pixel1[1]] in nearby and obj1 != obj:
                        if obj < obj1:
                            combines[obj] = obj1
                        else:
                            combines[obj1] = obj

    if combines.keys():
        print 'combines =', combines
        for k, v in combines.iteritems():
            for pix in objects[v]:
                objects[k].append(pix)
            objects.pop(v, None)
	'''

    return objects


def checkIfInsideObject(x, y, nearby, objects, pixels):
    #obj = -1
    #print 'in checkIfInsideObject, x, y, nearby =', x, y, nearby
    for obj in nearby:
        #print 'obj =', obj, nearby[obj]
        if [x, y] in nearby[obj]:
            #print 'checkIfInsideObject check for nearby returning ', obj
            return obj

    nearby = getNearbyPixels(x, y, pixels)
    for obj in objects:
        #print 'checkIfInsideObject checking for last row of obj', obj, objects[obj]
        for x1, y1 in objects[obj]:
            if [x1, y1] in nearby:
        #if [x, y] in objects[obj][-1]:
                #print 'checkIfInsideObject checking last row', objects[obj][-1]
                #print 'checkIfInsideObject check for last row returning ', obj
                return obj

    #print 'checkIfInsideObject returning -1'
    return -1

def getNearbyPixels(x, y, pixels):
    nearby = []
    for x1, y1 in [[x-1, y], [x-1, y-1],
                   [x, y-1], [x, y+1],
                   [x+1,y+1], [x+1, y],
                   [x-1, y+1], [x+1, y-1]]:
        if pixels[x1][y1] == 0:
            nearby.append([x1, y1])
    #print 'returning from getNearbyPixels, nearby =', nearby
    return nearby



def checkForMovementType(edge):
    solution = -1
    imgA = moveImageToEdge('A', edge)
    imgB = moveImageToEdge('B', edge)
    imgC = moveImageToEdge('C', edge)
    AB = moveImageToEdge(xor2Images(imgA, imgB), edge)
    imgD = moveImageToEdge('D', edge)
    imgE = moveImageToEdge('E', edge)
    imgF = moveImageToEdge('F', edge)
    DE = moveImageToEdge(xor2Images(imgD, imgE), edge)
    if checkForEquality(AB, imgC) and checkForEquality(DE, imgF):
        print 'type is moveImageToEdge', edge
        imgG = moveImageToEdge('G', edge)
        imgH = moveImageToEdge('H', edge)
        GH = moveImageToEdge(xor2Images(imgG, imgH), edge)
        for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
            imgI = moveImageToEdge(i, edge)
            if checkForEquality(GH, imgI):
                solution = int(i)
            if solution != -1:
                return solution
    return -1


def moveImageToEdge(i, edge):
    if isinstance(i, basestring):
        img = Images1[i]
    else:
        img = i
    #Images1['A'].show()
    firstBlack = getFirstXY(i, edge)
    #print 'firstBlack =', firstBlack
    img = ImageChops.offset(img, -1 * firstBlack[0], 0)
    return img

def getFirstXY(i, edge):
    if isinstance(i, basestring):
        img = Images1[i]
    else:
        img = i
    imgData = img.getdata()
    pixels = img.load()
    #print pixels[92, 150]
    maxxy = img.size
    #matrix = [[0 for y in range(maxxy[1])] for x in range(maxxy[0])]
    rangex = range(maxxy[0])
    if edge == 'right':
        rangex = reversed(range(maxxy[0]))
    for x in rangex:
        for y in range(maxxy[1]):
            if pixels[x, y] == 0:
                firstBlack = [x, y]
                return firstBlack
    return [0,0]

def getLastXY(i):
    if isinstance(i, basestring):
        img = Images1[i]
    else:
        img = i
    imgData = img.getdata()
    pixels = img.load()
    #print pixels[92, 150]
    maxxy = img.size
    #matrix = [[0 for y in range(maxxy[1])] for x in range(maxxy[0])]
    for x in reversed(range(maxxy[0])):
        for y in reversed(range(maxxy[1])):
            if pixels[x, y] == 0:
                firstBlack = [x, y]
                return firstBlack
    return [0,0]

    #for pixel in imgData:
        #print pixel



def fillHalfImage(i1, half):
    ax1, ay1, ax2, ay2 = Images1[i1].getbbox()
    print 'ax1, ay1, ax2, ay2 =', ax1, ay1, ax2, ay2
    i1Copy= Images[i1].convert('1')
    if half == 'top':
        coords = (ax1, ay1+(ay2-ay1)/2 + 1, ax2, ay2)
    if half == 'bottom':
        coords = (ax1, ay1, ax2, ay2/2)
    # left half = firsthalf = (ax1+8, ay1, ax2/2, ay2)
    # right half = secondhalf = (ax1+(ax2-ax1)/2 + 1, ay1, ax2-7, ay2)
    print 'coords = ', coords

    #i1Copy.paste((255,255,255), coords)
    i1Copy.paste(255, coords)

    return i1Copy

def calcRatio(a, b):
    ratio = a / b
    if ratio > 1.0:
        return 1.0 / ratio
    return ratio

def getDPR(i1):
    if isinstance(i1, basestring):
        img1 = Images1[i1]
    else:
        img1 = i1
    #img1 = rgbImages[i1]
    img1colors = img1.getcolors()
    #print 'colors =', img1colors
    img1BlackCount = 0
    img1WhiteCount = 0
    for pixelCount, rgb in img1colors:
        #print 'type of pixelCount, rgb = ', type(pixelCount), type(rgb)
        #if rgb < (128, 128, 128):
        if rgb == 0: #(0, 0, 0):
            #print 'rgb = ', rgb
            #print 'pixelCount, rgb = ', pixelCount, rgb
            img1BlackCount = float(pixelCount)
        else:
            img1WhiteCount = float(pixelCount)
    #print 'img1BlackCount, img1WhiteCount = ', img1BlackCount, img1WhiteCount
    #print 'dpr = ', img1BlackCount/(img1BlackCount + img1WhiteCount)
    dpr = round(img1BlackCount/(img1BlackCount + img1WhiteCount), 2)

    return img1BlackCount, img1WhiteCount, dpr

def min2Images(i1, i2):
    img3 = ImageMath.eval("convert(min(a, b), '1')", a=Images1[i1], b=Images1[i2])
    return img3

def subtract2Images(i1, i2):
    img3 = ImageMath.eval("convert(a-b, '1')", a=Images1[i1], b=Images1[i2])
    img3 = ImageChops.invert(img3)
    return img3

def multiply2Images(i1, i2):
    if isinstance(i1, basestring):
        img1 = Images1[i1]
    else:
        img1 = i1
    if isinstance(i2, basestring):
        img2 = Images1[i2]
    else:
        img2 = i2
    img3 = ImageMath.eval("convert(a*b, '1')", a=img1, b=img2)
    #img3 = ImageChops.invert(img3)
    return img3

def add2Images(i1, i2):
    img3 = ImageMath.eval("convert(a+b, '1')", a=Images1[i1], b=Images1[i2])
    #img3 = ImageChops.invert(img3)
    return img3

def divide2Images(i1, i2):
    img3 = ImageMath.eval("convert(a/b, '1')", a=Images1[i1], b=Images1[i2])
    #img3 = ImageChops.invert(img3)
    return img3

def max2Images(i1, i2):
    img3 = ImageMath.eval("convert(max(a, b), '1')", a=Images1[i1], b=Images1[i2])

    return img3

def xor2Images(i1, i2):
    if isinstance(i1, basestring):
        img1 = Images1[i1]
    else:
        img1 = i1
    if isinstance(i2, basestring):
        img2 = Images1[i2]
    else:
        img2 = i2
    img3 = ImageChops.logical_xor(img1, img2)
    img3 = ImageChops.invert(img3)
    return img3

def or2Images(i1, i2):
    if isinstance(i1, basestring):
        img1 = Images1[i1]
    else:
        img1 = i1
    if isinstance(i2, basestring):
        img2 = Images1[i2]
    else:
        img2 = i2

    img3 = ImageChops.logical_or(img1, img2)
    #img3 = ImageChops.invert(img3)
    return img3

def xor3Images(i1, i2, i3):
    img3 = xor2Images(Images1[i1], Images1[i2])
    img4 = xor2Images(Images1[i2], Images1[i3])
    img5 = xor2Images(img3, img4)
    return img5

def or3Images(i1, i2, i3):
    img3 = or2Images(Images1[i1], Images1[i2])
    img4 = or2Images(Images1[i2], Images1[i3])
    img5 = or2Images(img3, img4)
    return img5

def union2Images(i1, i2):
    if isinstance(i1, basestring):
        img1 = Images1[i1]
    else:
        img1 = i1
    if isinstance(i2, basestring):
        img2 = Images1[i2]
    else:
        img2 = i2

    img3 = ImageChops.logical_and(img1, img2)
    return img3

def union3Images(i1, i2, i3):
    img3 = ImageChops.logical_and(Images1[i1], Images1[i2])
    img4 = ImageChops.logical_and(img3, Images1[i3])
    return img4



def getDPRRatio(imageSet):
    i1 = imageSet[0]
    i2 = imageSet[1]
    i3 = imageSet[2]
    dprRatio = 0.0

    #print 'calculating dark pixel ratio for images', i1, i2, i3

    img1BlackCount, img1WhiteCount, dpr1 = getDPR(i1)
    img2BlackCount, img2WhiteCount, dpr2 = getDPR(i2)
    img3BlackCount, img3WhiteCount, dpr3 = getDPR(i3)

    diffBlack12 = img2BlackCount - img1BlackCount
    diffBlack23 = img3BlackCount - img2BlackCount
    diffRatio1223 = diffBlack12 / (diffBlack23 + 0.00001)

    #print 'diffBlack12, diffBlack23', diffBlack12, diffBlack23
    #print 'diffRatio1223 =', diffRatio1223
    if diffRatio1223 > 1.0:
        diffRatio1223 = 1.0/diffRatio1223
    #print 'diffRatio1223 =', diffRatio1223

    return diffRatio1223

def checkForCount(set1, set2):
    diffRatioABBC = getDPRRatio(set1)
    diffRatioDEEF = getDPRRatio(set2)

    #print 'diffRatioABBC, diffRatioDEEF =', diffRatioABBC, diffRatioDEEF

    ratioOfRatios = diffRatioABBC / diffRatioDEEF
    if ratioOfRatios > 1.0:
        ratioOfRatios = round(1.0/ratioOfRatios, 2)
    #print 'ratioOfRatios = ', ratioOfRatios
    if ratioOfRatios >= 0.9:
        return ratioOfRatios, True

    '''diff = abs(abs(dpr2 - dpr1) - abs(dpr3 - dpr2))
    print 'abs(abs(dpr2 - dpr1) - abs(dpr3 - dpr2))', diff
    if diff <= 0.02:
        #return True

        ratio = abs(dpr2 - dpr1) / abs(dpr3 - dpr2 + 0.0001)
        print 'abs(dpr2 - dpr1) / abs(dpr3 - dpr2 + 0.0001) = ', ratio
        if ratio >= 0.96:
            return True'''
    return ratioOfRatios, False

def checkForEquality(i1, i2):
    img1 = i1
    img2 = i2    
    if isinstance(i1, basestring) and isinstance(i2, basestring):
        img1 = Images1[i1]
        img2 = Images1[i2]
    elif isinstance(i1, basestring) and not isinstance(i2, basestring):
        if img2.getbands() == ('1',):
            img1 = Images1[i1]
        else:
            img1 = rgbImages[i1]
    elif isinstance(i2, basestring) and not isinstance(i1, basestring):
        if img1.getbands() == ('1',):
            img2 = Images1[i2]
        else:
            img2 = rgbImages[i2]
    
    if exactlyEqual(img1, img2):
        #print 'images ', i1, ' and', i2, ' exactly equal.'
        return True
    elif almostEqual(img1, img2):
        # diff = rmsdiff(Images['A'], Images['B'])  # rms diff does not work - returns 0
                                                    # even for images rotated/reflected
        #print 'images ', i1, ' and', i2, ' almost equal.'
        return True
    return False

# Taken from: http://effbot.org/zone/pil-comparing-images.htm
def exactlyEqual(img1, img2):
    diff = ImageChops.difference(img1, img2).getbbox()
    if diff is None:
        return True
    else:
        return False


def almostEqual(img1, img2):
    img1Data = img1.getdata()
    img2Data = img2.getdata()
    img1BlackCount = 0
    img2BlackCount = 0
    same = 0
    black = None
    if img1.getbands() == ('1',): # getbands return ('1',) for bilevel images
        black = 0
    else:
        black = (0, 0, 0)
    #print 'comparing for almost equal', i1, i2
    img1colors = img1.getcolors()
    img2colors = img2.getcolors()
    #print 'img1bands, black = ', img1.getbands(), black
    for pixelCount, rgb in img1colors:
        #print 'type of pixelCount, rgb = ', type(pixelCount), type(rgb)
        if rgb == black: #(0, 0, 0):
            #print 'pixelCount, rgb = ', pixelCount, rgb
            img1BlackCount = pixelCount
    for pixelCount, rgb in img2colors:
        if rgb == black: #(0, 0, 0):
            #print 'pixelCount, rgb = ', pixelCount, rgb
            img2BlackCount = pixelCount

    for pix1, pix2 in zip(img1Data, img2Data):
        #print 'pix1, pix2 = ', pix1, pix2
        if pix1 == black: #(0, 0, 0):
            if pix1 == pix2:
                same += 1
    #print 'img1BlackCount , img2BlackCount, same = ', img1BlackCount, img2BlackCount, same
    if abs(img1BlackCount - img2BlackCount) <= SIMILARITY_THRESHOLD and \
                    img1BlackCount - same <= SIMILARITY_THRESHOLD and \
                            img2BlackCount - same <= SIMILARITY_THRESHOLD:
        #print 'returning from SIMILARITY_THRESHOLD check'
        return True

    if float(same)/(float(img1BlackCount) + 0.00001) >= 0.939 and float(same)/(float(img2BlackCount) + 0.00001) >= 0.939:
        #print 'returning from percentage check'
        return True

    return False

def checkForAdds(i1, i2):
    img1 = Images1[i1]
    img2 = Images1[i2]
    img3 = ImageChops.logical_and(img1, img2)
    #print 'img1 bands = ', img1.getbands()
    #print 'img2 bands = ', img1.getbands()
    #print 'img3 bands = ', img1.getbands()
    #img3.show()
    #for i in img3.getdata():
    #    print i
    #print 'comapring i1 and AandB'
    return checkForEquality(img2, img3)
    
def checkForDeletes(i1, i2):
    img1 = Images1[i1]
    img2 = Images1[i2]
    img3 = ImageChops.logical_or(img1, img2)
    #print 'img1 bands = ', img1.getbands()
    #print 'img2 bands = ', img1.getbands()
    #print 'img3 bands = ', img1.getbands()
    #img3.show()
    #for i in img3.getdata():
    #    print i
    #print 'comapring i1 and AandB'
    return checkForEquality(img2, img3)

def checkForReflection(i1, i2, reflectionType):
    img1 = rgbImages[i1]
    img2 = rgbImages[i2]
    reflected = None

    if reflectionType == 'h':
        reflected = img2.transpose(Image.FLIP_LEFT_RIGHT)

    if reflectionType == 'v':
        reflected = img2.transpose(Image.FLIP_TOP_BOTTOM)
    return checkForEquality(img1, reflected)

# Taken from http://snipplr.com/view/757/
# Works
def rmsdiff(im1, im2):
    h1 = im1.histogram()
    h2 = im2.histogram()

    rms = math.sqrt(reduce(operator.add, map(lambda a, b: (a - b) ** 2, h1, h2)) / len(h1))

    return rms

# Taken from: http://effbot.org/zone/pil-comparing-images.htm
# Does not work
def rmsdiff2(im1, im2):
    h = ImageChops.difference(im1, im2).histogram()
    print 'histogram h = ', h

    # calculate rms
    #return math.sqrt(sum(h*(i**2) for i, h in enumerate(h))) / (float(im1.size[0]) * im1.size[1]))
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(len(h)))
    ) / (float(im1.size[0]) * im1.size[1]))

def identifySolution(predictedSolution):
    solution = -1
    solutionImageList = ['1', '2', '3', '4', '5', '6']
    #print 'solutions ', solutionImageList
    for i in solutionImageList:
        if checkForEquality(predictedSolution, i):
            solution = i

    print 'solution = ', solution
    return solution

def openAllImages(problem):
    if problem.problemType == '2x2':
        imageList = ['A', 'B', 'C', '1', '2', '3', '4', '5', '6']
        solutionImageList = ['1', '2', '3', '4', '5', '6']
    if problem.problemType == '3x3':
        imageList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '1', '2', '3', '4', '5', '6', '7', '8']
        solutionImageList = ['1', '2', '3', '4', '5', '6', '7', '8']

    for i in imageList:
        Images[i] = Image.open(problem.figures[i].visualFilename)
        rgbImages[i] = Images[i].convert('RGB')
        Images1[i] = Images[i].convert('1')
        grayscaleImages[i] = Images[i].convert('L')


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an integer representing its
    # answer to the question: "1", "2", "3", "4", "5", or "6". These integers
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName() (as Strings).
    #
    # In addition to returning your answer at the end of the method, your Agent
    # may also call problem.checkAnswer(int givenAnswer). The parameter
    # passed to checkAnswer should be your Agent's current guess for the
    # problem; checkAnswer will return the correct answer to the problem. This
    # allows your Agent to check its answer. Note, however, that after your
    # agent has called checkAnswer, it will *not* be able to change its answer.
    # checkAnswer is used to allow your Agent to learn from its incorrect
    # answers; however, your Agent cannot change the answer to a question it
    # has already answered.
    #
    # If your Agent calls checkAnswer during execution of Solve, the answer it
    # returns will be ignored; otherwise, the answer returned at the end of
    # Solve will be taken as your Agent's answer to this problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):
        solution = -1
        openAllImages(problem)
        # print ImageChops.difference(Images['A'], Images['B']).getbbox()
        #if problem.name != 'Basic Problem E-09': # or problem.name == 'Basic Problem D-08' or problem.name == 'Basic Problem E-07' or problem.name == 'Basic Problem E-09':
        #    return -1
        print 'problem: ', problem.name
        predictedSolution = ''
        if problem.problemType == '2x2':
            print 'checking for equality of A and B'
            if checkForEquality('A', 'B'):  # Horizontal equality
                predictedSolution = 'C'
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for equality of A and C'
            if checkForEquality('A', 'C'):  # Vertical equality
                predictedSolution = 'B'
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for horizontal reflection of A and B'
            if checkForReflection('A', 'B', 'h'):  # Horizontal reflection
                print 'B is a horizontal reflection of A'
                predictedSolution = rgbImages['C'].transpose(Image.FLIP_LEFT_RIGHT)
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for horizontal reflection of A and C'
            if checkForReflection('A', 'C', 'h'):  # Horizontal reflection
                print 'C is a horizontal reflection of A'
                predictedSolution = rgbImages['B'].transpose(Image.FLIP_LEFT_RIGHT)
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for vertical reflection of A and C'
            if checkForReflection('A', 'C', 'v'):  # Vertical equality
                print 'C is a vertical reflection of A'
                predictedSolution = rgbImages['B'].transpose(Image.FLIP_TOP_BOTTOM)
                #if problem.name == 'Basic Problem B-06':
                    #predictedSolution.show()
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for fills between A and B'
            if checkForAdds('A', 'B'):
                print 'B is a filled version of A'
                for i in ['1', '2', '3', '4', '5', '6']:
                    if checkForAdds('C', i):
                        solution = int(i)
                if solution != -1:
                    return solution

            print 'checking for deletes between A and B'
            # deletes from A to B is same as adds from B to A
            if checkForAdds('B', 'A'):
                print 'A is a filled version of B'
                for i in ['1', '2', '3', '4', '5', '6']:
                    print 'comparing C and ', i
                    if checkForAdds(i, 'C'):
                        print 'C is similar to', i
                        solution = int(i)
                    if solution != -1:
                        return solution

            print 'checking for fills between A and C'
            if checkForAdds('A', 'C'):
                print 'C is a filled version of A'
                for i in ['1', '2', '3', '4', '5', '6']:
                    if checkForAdds('B', i):
                        solution = int(i)
                if solution != -1:
                    return solution
        # Method to find filled version - does not work
        #ImageDraw.floodfill(Images['A'], xy, (0, 0, 0), border=None)
        #mask = Images['A'].convert('L')
        #th=150 # the value has to be adjusted for an image of interest
        #mask = mask.point(lambda i: i < th and 255)
        #mask.show()
        #print 'mask = ', mask
        #for x in mask.getdata():
        #    print x

        # Method to find filled version - works but may not be the right method
        #Images['B'].filter(ImageFilter.FIND_EDGES).show()
        #for x in Images['A'].convert('L').getdata():
        #    print x
        #print 'A histogram = ', Images1['A'].getdata()


        if problem.problemType == '3x3':
            if checkForEquality('A', 'B') and checkForEquality('B', 'C'):
                if checkForEquality('D', 'E') and checkForEquality('E', 'F'):
                    print 'type is equality'
                    predictedSolution = 'H'
                    solution = identifySolution(predictedSolution)
                    if solution != -1:
                        return solution

            if checkForEquality('A', 'E'):
                print 'type is diagonal equality'
                predictedSolution = 'E'
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            if checkForReflection('A', 'C', 'h') and checkForReflection('D', 'F', 'h'):
                predictedSolution = Images1['G'].transpose(Image.FLIP_LEFT_RIGHT)
                print 'type is reflection'
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            if checkForEquality(union2Images('B', 'D'), Images1['E']) and \
                checkForEquality(union2Images('C', 'E'), Images1['F']) and \
                checkForEquality(union2Images('E', 'G'), Images1['H']):
                print 'type is union of B and D'
                predictedSolution = union2Images('F', 'H')
                #print 'type of predictedSolution = ', type(predictedSolution)
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(predictedSolution, i):
                        solution = int(i)
                    if solution != -1:
                        return solution

            # TODO - this solves D-05 - may be incorrect
            if checkForEquality(xor2Images('A', 'B'), xor2Images('D', 'E')) and \
                    checkForEquality(xor2Images('B', 'C'), xor2Images('E', 'F')): # and \
                    #checkForEquality(xor2Images('A', 'B'), xor2Images('G', 'H')):
                print 'type is xor of A and B'
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(xor2Images('H', i), xor2Images('E', 'F')):
                        solution = int(i)
                        return solution


            ratio, countType = checkForCount(['A', 'B', 'C'], ['D', 'E', 'F'])
            if countType:
                print 'type is count'
                maxRatio = 0.0
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    ratio1, countType1 = checkForCount(['D', 'E', 'F'], ['G', 'H', i])
                    if countType1:
                        print 'i, ratio1, maxRatio = ', i, ratio1, maxRatio
                        if ratio1 > maxRatio:
                            solution = int(i)
                            maxRatio = ratio1
                        #print 'solution for count =', i
                if (solution != -1):
                    return solution


            # TODO - checkForEquality needs to return points
            # Ex: E-03 solution is both 1 and 2 if we dont use points
            if checkForEquality(union2Images('A', 'B'), Images1['C']) and \
                checkForEquality(union2Images('D', 'E'), Images1['F']):
                print 'type is union of A and B'
                predictedSolution = union2Images('G', 'H')
                #print 'type of predictedSolution = ', type(predictedSolution)
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(predictedSolution, i):
                        solution = int(i)
                if solution != -1:
                    return solution

            if checkForEquality(xor2Images('A', 'B'), Images1['C']) and \
                checkForEquality(xor2Images('D', 'E'), Images1['F']):
                print 'type is xor of AB = C'
                predictedSolution = xor2Images('G', 'H')
                #print 'type of predictedSolution = ', type(predictedSolution)
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(predictedSolution, i):
                        solution = int(i)
                if solution != -1:
                    return solution

            if checkForEquality(or2Images('A', 'B'), Images1['C']) and \
                checkForEquality(or2Images('D', 'E'), Images1['F']):
                print 'type is or of A and B'
                predictedSolution = or2Images('G', 'H')
                #print 'type of predictedSolution = ', type(predictedSolution)
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(predictedSolution, i):
                        solution = int(i)
                if solution != -1:
                    return solution


            if abs(getDPR('C')[0] - 2 * getDPR('A')[0]) < 110 and abs(getDPR('F')[0] - 2 * getDPR('D')[0]) < 110:
                print 'C is twice of A, F is twice of D'
                GPixels = getDPR('G')[0]

                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    pixels = getDPR(i)[0]
                    #print 'pixels =', pixels
                    diff = abs(pixels - 2 * GPixels)
                    #print 'diff =', diff
                    if diff < 100:
                        #print 'GPixels, pixels =', GPixels, pixels
                        solution = int(i)
                if (solution != -1):
                    return solution

            AandC = ImageChops.logical_and(Images1['A'], Images1['C'])
            DandF = ImageChops.logical_and(Images1['D'], Images1['F'])
            #AandC = AandC.filter(ImageFilter.FIND_EDGES).convert('1')
            if abs(getDPR(AandC)[0] - 2 * getDPR('B')[0]) < 100 and abs(getDPR(DandF)[0] - 2 * getDPR('E')[0]):
                print 'type is movement'
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    GandI = ImageChops.logical_and(Images1['G'], Images1[i])
                    if abs(getDPR(GandI)[0] - 2 * getDPR('H')[0]) < 100:
                        solution = int(i)
                if solution != -1:
                    return solution

            solution = checkForMovementType('left')
            if solution != -1:
                return solution

            solution = checkForMovementType('right')
            if solution != -1:
                return solution

            # TODO - moving set ABC = DEF calculation to after ADG = BEH gives incorrect result
            # TODO - this identifies E-12 also, to be fixed
            dprABC = getDPR('A')[0] + getDPR('B')[0] + getDPR('C')[0]
            dprDEF = getDPR('D')[0] + getDPR('E')[0] + getDPR('F')[0]
            r = calcRatio(dprABC, dprDEF)
            if (calcRatio(dprABC, dprDEF)) > 0.98:
                print 'type is set of ABC = DEF'
                maxRatio = 0.0
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    dprGHI = getDPR('G')[0] + getDPR('H')[0] + getDPR(i)[0]
                    # TODO - changing dprDEF to dprABC gives incorrect result - to be fixed
                    ratio = calcRatio(dprDEF, dprGHI)
                    #print 'dprADG, dprBEH, dprCFI, ratioABCDEF, ratioABCGHI =', dprABC, dprDEF, dprGHI, r, ratio
                    if ratio > 0.98:
                        if ratio > maxRatio:
                            solution = int(i)
                            maxRatio = ratio

                if solution != -1:
                    return solution

            dprADG = getDPR('A')[0] + getDPR('D')[0] + getDPR('G')[0]
            dprBEH = getDPR('B')[0] + getDPR('E')[0] + getDPR('H')[0]
            if (calcRatio(dprADG, dprBEH)) > 0.98:
                print 'type is set of ADG = BEH'
                maxRatio = 0.0
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    dprCFI = getDPR('C')[0] + getDPR('F')[0] + getDPR(i)[0]
                    #print 'dprADG, dprBEH, dprCFI =', dprADG, dprBEH, dprCFI
                    ratio = calcRatio(dprADG, dprCFI)

                    if (calcRatio(dprADG, dprCFI)) > 0.98:
                        if ratio > maxRatio:
                            solution = int(i)
                            maxRatio = ratio
                        #print 'dprADG, dprBEH, dprCFI =', dprADG, dprBEH, dprCFI
                if solution != -1:
                    return solution

            img3 = xor2Images(max2Images('A', 'B'), min2Images('A', 'B'))
            img4 = xor2Images(max2Images('D', 'E'), min2Images('D', 'E'))
            if checkForEquality(img3, Images1['C']) and checkForEquality(img4, Images1['F']):
                print 'type is MinABxorMaxAB = C'
                GH = xor2Images(min2Images('G', 'H'), max2Images('G', 'H'))
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(GH, Images1[i]):
                        solution = int(i)
                if solution != -1:
                    return solution

            AxorBxorC =  xor3Images('A', 'B', 'C')
            DxorExorF = xor3Images('D', 'E', 'F')

            if checkForEquality(AxorBxorC, DxorExorF):
                print 'type is AxorBxorC = DxorExorF'
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    GxorHxorI = xor3Images('G', 'H', i)
                    #CandFandI.show()
                    if checkForEquality(GxorHxorI, AxorBxorC):
                        #CandFandI.show()
                        solution = int(i)
                if (solution != -1):
                    return solution

            iACopy = fillHalfImage('A', 'top')
            iBCopy = fillHalfImage('B', 'bottom')
            iDCopy = fillHalfImage('D', 'top')
            iECopy = fillHalfImage('E', 'bottom')

            if checkForEquality(union2Images(iACopy, iBCopy), Images1['C']) and \
                    checkForEquality(union2Images(iDCopy, iECopy), Images1['F']):
                print 'E-09'
                iGCopy = fillHalfImage('G', 'top')
                iHCopy = fillHalfImage('H', 'bottom')
                #union2Images(iGCopy, iHCopy).show()
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    if checkForEquality(union2Images(iGCopy, iHCopy), Images1[i]):
                        solution = int(i)
                if solution != -1:
                    return solution

            img3 = union3Images('A', 'B', 'C')
            img4 = union3Images('D', 'E', 'F')
            #img3.show()
            #img4.show()
            if checkForEquality(img3, img4):
                print 'type is AunionBunionC = DunionEunionF'
                for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    img5 = union3Images('G', 'H', i)
                    #img5.show()
                    if checkForEquality(img4, img5):
                        solution = int(i)
                if solution != -1:
                    return solution

            #if checkForEquality(xor2Images(Images1['B'], Images1['E']), Images1['H']):
            #    print 'type'

            imgObjects = {}
            objCounts = {}
            pixelCounts = {}
            for img in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '1', '2', '3', '4', '5', '6', '7', '8']:
                pixelCounts[img] = 0.0
                imgObjects[img] = getObjectsPixelData(Images1[img])
                #print 'Image =', img
                #print 'Number of objects = ', len(imgObjects[img].keys())
                objCounts[img] = len(imgObjects[img].keys())
                for obj in imgObjects[img].keys():
                    pixelCounts[img] += len(imgObjects[img][obj])
                    #print 'len of obj', obj, len(imgObjects[img][obj])

            solution = checkObjectsCountType(objCounts, pixelCounts)
            if solution != -1:
                return solution

        return solution


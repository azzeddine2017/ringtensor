/*
    Project: Jibrail / Adam Engine Demo
    Author: Azzeddine Remmal & Code Gear-1
    Description: High-Performance 3D Rotation using RingTensor (GPU/CPU Optimized)
    Target: Render 12,890 particles in real-time.
*/

load "guilib.ring"
load "ringml.ring" 

? "num cores :" + tensor_get_cores()
# Enable Multi-Core
tensor_set_threads(2) 
# Set GPU Threshold
setGpuThreshold(100000)

Decimals(4)

Qt_AlignCenter = 1
oWin = null

# --- Configuration ---
cMugFile = "mug.txt"
nWidth   = 800
nHeight  = 700
nScale   = 2.0   # 5.0     
nSpeed   = 2.0   # 10 is too fast     



func main
    new qApp {
        StyleFusion()
        
        if !fexists(cMugFile) 
            msginfo("Error", "File mug.txt not found!") 
            return 
        ok

        oWin = new MainWindow("oWin")
        oWin.show()
        exec()
    }

class MainWindow from QWidget

    # Data
    oPoints         
    oTransformed    
    nPointsCount    = 0
    oObjectName     
    
    # Rotation
    nAngleX = 0
    nAngleY = 0
    nAngleZ = 0
    
    # Metrics
    nMathTime = 0
    
    # GUI
    oLabelImage
    oTimer
    oPixmap
    oPainter
    oPen
    oColorBlack
    oColorCyan
    
    func init ObjectName
        oObjectName = ObjectName
        super.init()
        
        setWindowTitle("RingTensor + drawHSVFList | Turbo Mode")
        setGeometry(100, 100, nWidth, nHeight)
        setStyleSheet("background-color: black;")
        
        oLabelImage = new QLabel(self) {
            setGeometry(0, 0, nWidth, nHeight)
            setAlignment(Qt_AlignCenter)
        }
        
        # Pre-allocate resources
        oPixmap     = new QPixmap2(nWidth, nHeight)
        oPainter    = new QPainter()
        
        # We don't need Pen/Color for HSVFList as it has its own colors, 
        # but we need background clear color
        oColorBlack = new QColor() { setRGB(0, 0, 0, 255) }
        
        loadMugData()
        
        oTimer = new QTimer(self)
        oTimer.setInterval(0) # 0 means as fast as possible
        oTimer.setTimeOutEvent(oObjectName + ".animate()")
        oTimer.start()

    func loadMugData
        see "Loading Data..." + nl
        cData = read(cMugFile)
        aLines = str2list(cData)
        nPointsCount = len(aLines)
        
        oPoints = new Tensor(nPointsCount, 4)
        
        for i = 1 to nPointsCount
            cLine = trim(aLines[i])
            if len(cLine) = 0 loop ok
            aParts = split(cLine, " ")
            aCoords = []
            for p in aParts if len(trim(p)) > 0 aCoords + number(p) ok next
            
            if len(aCoords) >= 3
                oPoints.setVal(i, 1, aCoords[1])
                oPoints.setVal(i, 2, aCoords[2])
                oPoints.setVal(i, 3, aCoords[3])
                oPoints.setVal(i, 4, 1.0)
            ok
        next
        
        oTransformed = new Tensor(nPointsCount, 4)
        see "Ready: " + nPointsCount + " points." + nl

    func animate
        nAngleX += nSpeed
        nAngleY += nSpeed
        nAngleZ += nSpeed
        if nAngleX > 360 nAngleX -= 360 ok
        if nAngleY > 360 nAngleY -= 360 ok
        if nAngleZ > 360 nAngleZ -= 360 ok
        
        t1 = clock()
        
        # 1. Math (C Engine)
        oRot = getRotationMatrix(nAngleX, nAngleY, nAngleZ)
        oTransformed = oPoints.matmul(oRot)

        # --- STRESS TEST: FORCE GPU LOAD ---
        # We will repeat the process 5000 times to see if the device will suffocate or not.
        /*
        for k = 1 to 5000
            oTransformed = oPoints.matmul(oRot)
        next
        */
        #-----------------------------------
        # 2. Extract Data (C to Ring List)
        aFlatData = oTransformed.toList()
        
        t2 = clock()
        nMathTime = (t2 - t1) / clockspersecond()
        
        # 3. Render
        draw_frame(aFlatData)

    func draw_frame aData
        
        # Reset Canvas
        oPixmap.fill(oColorBlack)
        
        # Prepare Batch List for drawHSVFList
        # Format per point: [x, y, h, s, v, alpha]
        aBatchList = []
        
        cx = nWidth / 2
        cy = nHeight / 2
        scale = nScale
        
        # HSV Color for Cyan-ish look
        h = 0.5  # Hue
        s = 1.0  # Saturation
        v = 1.0  # Value
        a = 1.0  # Alpha
        
        nLen = len(aData)
        
        # --- PROJECTION LOOP ---
        # Although this loop is in Ring, building a list is much faster 
        # than calling oPainter.drawPoint() 13,000 times.
        
        for i = 1 to nLen step 4
            x = aData[i]
            y = aData[i+1]
            z = aData[i+2]
            
            # Simple Perspective Projection
            dist = 200
            if (z + dist) != 0
                factor = 200.0 / (z + dist)
                
                px = (x * scale * factor) + cx
                py = (y * scale * factor) + cy
                
                # Check Bounds (Optimization)
                if px >= 0 and px < nWidth and py >= 0 and py < nHeight
                    # Add to batch list
                    aBatchList + [px, py, h, s, v, a]
                ok
            ok
        next
        
        # --- BATCH DRAW (The Fast Part) ---
        oPainter.begin(oPixmap)
        
        # Draw Statistics
        oPainter.setPen(new QPen() { setColor(new QColor(){setRGB(255,255,255,255)}) })
        oPainter.drawText(20, 30, "Points: " + nPointsCount)
        oPainter.drawText(20, 50, "Math Time: " + nMathTime + "s")
        if nMathTime > 0
            oPainter.drawText(20, 70, "FPS (Calc): " + floor(1/nMathTime))
        ok
        
        # Draw All Points at Once
        oPainter.drawHSVFList(aBatchList)
        
        oPainter.endpaint()
        
        # Show Result
        oLabelImage.setPixmap(oPixmap)

    # --- Matrix Helpers ---
    func getRotationMatrix ax, ay, az
        # Convert to Radians
        rad = 3.1415926535 / 180
        rx = ax * rad
        ry = ay * rad
        rz = az * rad
        
        # Matrix Rx
        # 1  0  0  0
        # 0  c -s  0
        # 0  s  c  0
        # 0  0  0  1
        oRx = new Tensor(4, 4)
        oRx.setVal(1,1, 1) oRx.setVal(4,4, 1)
        c=cos(rx) s=sin(rx)
        oRx.setVal(2,2, c) oRx.setVal(2,3, 0-s)
        oRx.setVal(3,2, s) oRx.setVal(3,3, c)
        
        # Matrix Ry
        # c  0  s  0
        # 0  1  0  0
        # -s 0  c  0
        # 0  0  0  1
        oRy = new Tensor(4, 4)
        oRy.setVal(2,2, 1) oRy.setVal(4,4, 1)
        c=cos(ry) s=sin(ry)
        oRy.setVal(1,1, c) oRy.setVal(1,3, s)
        oRy.setVal(3,1, 0-s) oRy.setVal(3,3, c)
        
        # Matrix Rz
        # c -s  0  0
        # s  c  0  0
        # 0  0  1  0
        # 0  0  0  1
        oRz = new Tensor(4, 4)
        oRz.setVal(3,3, 1) oRz.setVal(4,4, 1)
        c=cos(rz) s=sin(rz)
        oRz.setVal(1,1, c) oRz.setVal(1,2, 0-s)
        oRz.setVal(2,1, s) oRz.setVal(2,2, c)
        
        # Combine: R = Rx * Ry * Rz
        # This is fast because 4x4 is tiny
        oRxy = oRx.matmul(oRy)
        oFinal = oRxy.matmul(oRz)
        
        return oFinal
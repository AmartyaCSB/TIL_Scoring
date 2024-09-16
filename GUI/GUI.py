# Full Script for GUI 

# Advanced zoom example. Like in Google Maps.
# It zooms only a tile, but not the whole image. So the zoomed tile occupies
# constant memory and not crams it with a huge resized image for the large zooms.
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # self.grid_remove()
            pass
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)


    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

class Zoom_Advanced(ttk.Frame):
    ''' Advanced zoom of the image '''
    def __init__(self, mainframe, image):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        # self.master.title('Zoom with mouse wheel')
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,width=600,height=500,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='') # nswe
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.image = Image.fromarray(image)  # open image
        self.width, self.height = self.image.size
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        self.show_image()

    def scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale        /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale        *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()

    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

def opennew():
    newfile = askopenfilename(title="Select new image to Process")

def openold():
    oldfile = askopenfilename(title="Select processed image to open")

def updateImage(alpha=0.7):
    inputimage = background[:]
    beta = 1-alpha
    x,y,c = np.shape(inputimage)
    secondimage = np.zeros((x,y,3)).astype(np.uint8)
    # print([LymphFlag.get(), InvTumFlag.get(), TumStromaFlag.get(), InSituTumorFlag.get(), HealthyFlag.get(), InflamFlag.get(), RestFlag.get()])
    masks = 0

    if LymphFlag.get() == 1:
        masks += 1
    if InvTumFlag.get() == 1:
        secondimage = secondimage + lab1[:,:,0:3].astype(np.uint8)
        masks += 1
    if TumStromaFlag.get() == 1:
        secondimage = secondimage + lab2[:,:,0:3].astype(np.uint8)
        masks += 1
    if InSituTumorFlag.get() == 1:
        secondimage = secondimage + lab3[:,:,0:3].astype(np.uint8)
        masks += 1
    if HealthyFlag.get() == 1:
        secondimage = secondimage + lab4[:,:,0:3].astype(np.uint8)
        masks += 1
    if NecrosisFlag.get() == 1:
        secondimage = secondimage + lab5[:,:,0:3].astype(np.uint8)
        masks += 1
    if InflamFlag.get() == 1:
        secondimage = secondimage + lab6[:,:,0:3].astype(np.uint8)
        masks += 1
    if RestFlag.get() == 1:
        secondimage = secondimage + lab7[:,:,0:3].astype(np.uint8)
        masks += 1

    if masks == 0:
        outimage = inputimage
    else:
        outimage = cv2.addWeighted(inputimage, 1, secondimage, beta, 0).astype(np.uint8)

    app.image = Image.fromarray(outimage)
    app.show_image()   
 
def IsolateTumor():
    LymphFlag.set(0)
    InvTumFlag.set(1)
    TumStromaFlag.set(1)
    InSituTumorFlag.set(1)
    HealthyFlag.set(0)
    NecrosisFlag.set(0)
    InflamFlag.set(0)
    RestFlag.set(0)
    updateImage()

def highlighTILcont():
    LymphFlag.set(1)
    InvTumFlag.set(1)
    TumStromaFlag.set(1)
    InSituTumorFlag.set(1)
    HealthyFlag.set(0)
    NecrosisFlag.set(0)
    InflamFlag.set(1)
    RestFlag.set(0)
    updateImage()

def ResetAll():
    LymphFlag.set(0)
    InvTumFlag.set(0)
    TumStromaFlag.set(0)
    InSituTumorFlag.set(0)
    HealthyFlag.set(0)
    NecrosisFlag.set(0)
    InflamFlag.set(0)
    RestFlag.set(0)
    updateImage()

if __name__ == "__main__":
    path = "D:/School/Spring2022/mip/GUI"
    lab1 = cv2.imread(path + "/images/invTumor_1.png",cv2.IMREAD_UNCHANGED)
    lab2 = cv2.imread(path + "/images/stroma_2.png", cv2.IMREAD_UNCHANGED)
    lab3 = cv2.imread(path + "/images/insitutumor_3.png",cv2.IMREAD_UNCHANGED)
    lab4 = cv2.imread(path + "/images/healthy_4.png",cv2.IMREAD_UNCHANGED)
    lab5 = cv2.imread(path + "/images/necrosis_5.png",cv2.IMREAD_UNCHANGED)
    lab6 = cv2.imread(path + "/images/inflamed_6.png",cv2.IMREAD_UNCHANGED)
    lab7 = cv2.imread(path + "/images/rest_7.png",cv2.IMREAD_UNCHANGED)
    background = cv2.imread(path + "/images/Tissue.png")

    # Setting up Main window and image
    root = tk.Tk()
    root.title('Humble Beginnings')
    root.resizable(False, False)
    root.geometry('820x590')

    # New file and open File subframe
    iframe_IO = tk.Frame(root, bd=1,relief=tk.SOLID)
    iframe_IO.grid(row=0,column=0,columnspan=3,sticky=tk.W,padx=4)

    # ------- Creating flags
    LymphFlag = tk.IntVar()
    InvTumFlag = tk.IntVar()
    TumStromaFlag = tk.IntVar()
    InSituTumorFlag = tk.IntVar()
    HealthyFlag = tk.IntVar()
    NecrosisFlag = tk.IntVar()
    InflamFlag = tk.IntVar()
    RestFlag = tk.IntVar()

    new_button = tk.Button(iframe_IO, text='New Image',font=("Arial",9),command=opennew).pack(side=tk.LEFT)
    open_button = tk.Button(iframe_IO, text='Open Project',font=("Arial",9),command=openold).pack(side=tk.RIGHT)

    #### Canvas subtitle frame
    iframe_canvas_title = tk.Frame(root)
    iframe_canvas_title.grid(row=1,column=0)
    label_canvas = tk.Label(iframe_canvas_title, width=20, text='Active Image',font=("Arial",16),).pack()

    #### Canvas Frame
    iframe_canvas= tk.Frame(root, bd=2,relief=tk.SOLID,width=500,height=500)
    iframe_canvas.grid(row=2,column=0,rowspan=3,padx=5)
   


    #### Simple Mask subframe
    iframe_mask = tk.Frame(root, bd=2,relief=tk.FLAT)
    iframe_mask.grid(row=2,column=1,rowspan=1,sticky=tk.N)
    # iframe_mask.pack(expand=1, fill=tk.Y, pady=10, padx=5)

    mask_label = tk.Label(iframe_mask, width=20, text='Single Layer masks',font=("Arial",12))
    mask_label.pack()

   

    # ------- Creating Checkboxes for simple masks
    cLymph = tk.Checkbutton(iframe_mask, text='Lymphocytes and Monocytes',font=("Arial",9),variable=LymphFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cInvTum = tk.Checkbutton(iframe_mask, text='Invasive Tumor',font=("Arial",9),variable=InvTumFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cStrom = tk.Checkbutton(iframe_mask, text='Tumor Stroma',font=("Arial",9),variable=TumStromaFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cISTum = tk.Checkbutton(iframe_mask, text='In-situ Tumor',font=("Arial",9),variable=InSituTumorFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cHealthy = tk.Checkbutton(iframe_mask, text='Healthy Tissue',font=("Arial",9),variable=HealthyFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cNecrosis = tk.Checkbutton(iframe_mask, text='Necrosis',font=("Arial",9),variable=NecrosisFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cInflam = tk.Checkbutton(iframe_mask, text='Inflamed Tissue',font=("Arial",9),variable=InflamFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)
    cRest = tk.Checkbutton(iframe_mask, text='Rest',font=("Arial",9),variable=RestFlag, onvalue=1, offvalue=0, command=updateImage).pack(anchor=tk.W)

    ### Preset-Masks Subframe
    iframe_preset = tk.Frame(root, bd=2,relief=tk.FLAT)
    iframe_preset.grid(row=3,column=1, sticky=tk.N)
    # iframe_preset.pack(expand=1, fill=tk.Y,side=tk.RIGHT, pady=10, padx=5)

    mask_label = tk.Label(iframe_preset, width=20, text='Preset Masks',font=("Arial",12))
    mask_label.pack()

    # Preset masks -  Creating Flags
    TumisolateFlag =tk.IntVar()
    TILcontFlag = tk.IntVar()

    cIsolateTum = tk.Button(iframe_preset, text='Isolate Tumor Bulk',font=("Arial",10),command=IsolateTumor,width=20).pack()
    cIsolateTIL = tk.Button(iframe_preset, text='Show TIL Contribution',font=("Arial",10),command=highlighTILcont,width=20).pack()
    cResetAll = tk.Button(iframe_preset, text='Reset all',font=("Arial",10),command=ResetAll,width=20).pack()

    # cIsolateTum = tk.Checkbutton(iframe_preset, text='Isolate tumor Bulk',variable=TumisolateFlag, onvalue=1, offvalue=0, command=IsolateTumor).pack(anchor=tk.W)
    # cIsolateTIL = tk.Checkbutton(iframe_preset, text='Isolate TILL Contribution',variable=TILcontFlag, onvalue=1, offvalue=0, command=IsolateTumor).pack(anchor=tk.W)
    # cResetAll = tk.Checkbutton(iframe_preset, text='Reset All',command=ResetAll).pack(anchor=tk.W)

    # Creating TIL window Subframe of preset subframe
    iframe_score = tk.Frame(root,bd=2,relief=tk.SOLID)
    iframe_score.grid(row=4,column=1,sticky=tk.S,pady = 4)
    # iframe_score.pack(fill=tk.X,pady=20,side=tk.BOTTOM)


    score_label = tk.Label(iframe_score,width=12, text='TIL Score: 70  ',font=("Arial",15)).pack(anchor=tk.W)

    app = Zoom_Advanced(iframe_canvas, background)

    
    # Executing GUI
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()
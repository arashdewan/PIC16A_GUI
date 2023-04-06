#!/usr/bin/env python
# coding: utf-8

# In[12]:



import tkinter as tk
import pickle


# FRAME OUTLINE 
window = tk.Tk() 
window.geometry("400x300") 
window.title("Penguin Classifier") 
window.minsize(width=300, height=300) 

frame_header = tk.Frame(window, 
                        borderwidth=2, 
                        pady=2, 
                        bg = "black")

frame_header.grid(row=0, column=0)

label_header = tk.Label(frame_header, 
                        text = "What kind of penguin are you??", 
                        bg='white', 
                        fg='black', 
                        height=2, width = 35,
                        font=("Helvetica 16 bold"))

label_header.grid(row=0, column=0)

frame_center = tk.Frame(window, 
                        borderwidth=2, 
                        pady=2, 
                        bg = "black")

frame_center.grid(row=1, column=0)

frame_bottom = tk.Frame(window, 
                        borderwidth=2, 
                        pady=2, 
                        bg = "black")

frame_bottom.grid(row=2, column=0)


# FIELDS FOR USER INPUT AND ANSWER

frame_main_1 = tk.Frame(frame_center,
                        borderwidth=2)
frame_main_2 = tk.Frame(frame_center,
                        borderwidth=2)
frame_main_3 = tk.Frame(frame_center,
                        borderwidth=2)

var1_label = tk.Label(frame_main_1, 
                      text = "Island: ")
var2_label = tk.Label(frame_main_2, 
                      text = "Body Mass (g): ")
var3_label = tk.Label(frame_main_3,
                      text = "Culmen Length (mm): ")

frame_main_1.pack(fill='x', pady=2)
frame_main_2.pack(fill='x', pady=2)
frame_main_3.pack(fill='x', pady=2)

var1_label.pack(side='left')
var2_label.pack(side='left')
var3_label.pack(side='left')

frame_answer = tk.Frame(frame_bottom, borderwidth=2, bg = "white")
frame_answer.pack(fill='x', pady=0)

# TEXT VARIABLES AND ENTRY 

var1 = tk.StringVar()
var1_entry = tk.OptionMenu(frame_main_1, 
                           var1, 
                           "Torgersen", 
                           "Biscoe", 
                           "Dream")

var1_entry.pack(side='right', 
                padx=1)

var2 = tk.StringVar()

var2_entry = tk.Entry(frame_main_2, 
                      textvariable = var2, 
                      width=10)
var2_entry.pack(side='right', 
                padx=1)

var3 = tk.StringVar()
var3_entry = tk.Entry(frame_main_3, 
                      textvariable = var3, 
                      width=10)
var3_entry.pack(side='right', 
                padx=1)

# INCORPORATING MACHINE LEARNING

T = pickle.load(open("model.p", "rb"))



island_dict = {
    "Biscoe"    : 0,
    "Dream"     : 1,
    "Torgersen" : 2
}
species_dict = {
    0 : "Adelie",
    1 : "Chinstrap",
    2 : "Gentoo"
}

answer_var = tk.StringVar()
def make_prediction():

    x = [[island_dict[var1.get()],
          float(var2.get()),
          float(var3.get())]]

    y_pred = species_dict[T.predict(x)[0]]

    ans = "You might be a(n) " + y_pred + " penguin."
    answer_var.set(ans)

button_run = tk.Button(frame_answer, 
                       text="Update", 
                       bg='blue', 
                       fg='black', 
                       relief='raised', 
                       width=10, 
                       font=('Helvetica 12 bold'),

command = make_prediction)

button_run.grid(column=0, 
                row=1, 
                sticky='w', 
                padx=100, 
                pady=2)

answer_label = tk.Label(frame_answer, 
                        font=('arial', 16, 'bold'),
                        bd=16, 
                        anchor="w",
                        textvariable=answer_var,
                        bg="white")  
 
answer_label.grid(row=0, column=0)

window.mainloop()


# In[ ]:





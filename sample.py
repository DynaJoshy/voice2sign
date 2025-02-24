import tkinter as tk

root = tk.Tk()
canvas = tk.Canvas(root, width=400, height=300, bg="white")
canvas.pack()

circle = canvas.create_oval(50, 50, 100, 100, fill="blue")

def move_circle():
    canvas.move(circle, 5, 0)  # Move 5 pixels to the right
    root.after(50, move_circle)  # Repeat every 50 ms

move_circle()
root.mainloop()

from tkinter import *
from tkinter import ttk, font

from os import system

# Main Window:

class main_menu():
    def __init__(self):
        self.root = Tk()
        self.root.geometry('1000x1000')
        self.root.configure(bg = 'black')
        self.root.title("Ghost'n OpenAI'n Goblins")
        self.keyfont = font.Font(size=40, weight='bold')
        self.button1 = Button(self.root, text='Exit',font=self.keyfont ,command=self.root.destroy).pack(side=BOTTOM, fill=BOTH, expand=True, padx=5, pady=5)
        self.button2 = Button(self.root, text='Play Game',font=self.keyfont, command=quit).pack(side=BOTTOM, fill=BOTH, expand=True, padx=5, pady=5)
        self.button3 = Button(self.root, text='AI Plays',font=self.keyfont, command=self.ia_play).pack(side=BOTTOM, fill=BOTH, expand=True, padx=5, pady=5)
        self.button4 = Button(self.root, text='Train AI',font= self.keyfont, command=self.train).pack(side=BOTTOM, fill=BOTH, expand=True, padx=5, pady=5)
        self.button4 = Button(self.root, text='Render Records to mp4',font= self.keyfont, command=self.render_records).pack(side=BOTTOM, fill=BOTH, expand=True, padx=5, pady=5)

        
        self.root.mainloop()
        
    def ia_play(self):
        _ = system('python3 ai_plays.py')

    def train(self):
        _= system('python3 paralel_train.py')

    def render_records(self):
        _ = system('python3 render_videos.py')

    def interactive(self):
        _ = system('python3 interactive.py')

        

    

    
def main():
    menu = main_menu()
    return 0


if __name__ == "__main__":
    main()

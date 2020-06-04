from shogi_engine import id2str,str2id,Game_state
import tkinter as tk
import tkinter.messagebox as msg
import ABPruning       
import random
from threading import Timer

class PromotionWindow(tk.Toplevel):
    #Window that pops up to ask you if you want to promote a piece or not
    def __init__(self,master):
        
        self.master = master
        tk.Toplevel.__init__(self)
        
    def disp(self):
                         
        self.geometry("+400+300")
        
        self.question = tk.Label(self, text = "Promouvoir ?")
        self.question.pack(padx = 5, pady = 15, side = tk.TOP)
                                
        self.buttonYes = tk.Button(self, text = "Oui", command = self.PromoteYes)
        self.buttonYes.pack(padx = 30, pady = 5, side = tk.LEFT)
                                
        self.buttonNo = tk.Button(self, text = "Non", command = self.destroy)
        self.buttonNo.pack(padx = 30, pady = 5, side = tk.RIGHT)
        
        self.transient(self.master)
        self.grab_set()
        self.master.wait_window(self)
        
    def PromoteYes(self):
        
        self.master.promotion = "+"
        self.destroy()
        
class SideWindow(tk.Toplevel):
    #Window that pops up to ask you which side you wish to play as
    def __init__(self,master):
        
        self.master = master
        tk.Toplevel.__init__(self)
        
    def disp(self):
                         
        self.geometry("+400+300")
        
        self.question = tk.Label(self, text = "Choisissez un côté")
        self.question.pack(padx = 5, pady = 15, side = tk.TOP)
                                
        self.buttonYes = tk.Button(self, text = "Blanc", command = self.white)
        self.buttonYes.pack(padx = 30, pady = 5, side = tk.RIGHT)
                                
        self.buttonNo = tk.Button(self, text = "Aléatoire", command = self.randomSide)
        self.buttonNo.pack(padx = 30, pady = 5, side = tk.RIGHT)
        
        self.buttonNo = tk.Button(self, text = "Noir", command = self.black)
        self.buttonNo.pack(padx = 30, pady = 5, side = tk.RIGHT)
        
        self.transient(self.master)
        self.grab_set()
        self.master.wait_window(self)
        
    def black(self):
        
        self.master.chosenSide = 'b'
        self.destroy()
        
    def randomSide(self):
        if random.random() < 1/2:
             self.master.chosenSide = 'b'
        else:
            self.master.chosenSide = 'w'
        self.destroy()
        
    def white(self):
        
        self.master.chosenSide = 'w'
        self.destroy()

class Board(tk.Canvas):
    
    def __init__(self,parent,pieces):
        # Initialiazing the board
        
        self.width = 452
        self.height = 452
        self.parent = parent
        self.highlighted = [] #Contains the highlighted positions
        tk.Canvas.__init__(self, master = parent, width = self.width, height = self.height, bg = "#FFE4B5", highlightthickness = 0)
        self.pieces = pieces #Array containing the pieces' positions
        #Initialization
        self.draw_pieces()
        
        
        
    def color_square(self,x,y,inside = "#FFE4B5"): 
        #Colors a square from the grid (x and y are the position in the matrix *50)
        self.create_rectangle(x+1,y+1,x+50,y+50,fill = inside)
            
            
    def highlight_reachable_squares(self,piece): 
        #Makes a list of all the squares that should be highlighted when we reload the board
        if piece == None:
            self.highlighted = []
        else:
            for move in gameState.legal_moves:
                if move[:2] == id2str(piece.pos):
                    self.highlighted.append(str2id(move[2:4])[::-1]) #Inverting pos is necessary because of the
                #way the board is drawn
                
            
    def draw_pieces(self): 
        #Draws the pieces, call each turn
        for i in range(9):
            for j in range(9):
                x = i*50
                y = j*50
                piece = self.pieces[j,i] #Counter-intuitive but it's actually the right way
                #We build the background of the piece
                if (i,j) in self.highlighted: #Blue background if reachable for the piece
                    self.color_square(x,y,"#0000FF") 
                else: #Classic background otherwise
                    self.color_square(x,y)                  
                if piece != None: #And then we draw the piece
                    piece.draw(self,x,y)
                    
class Captured(tk.Canvas):
    #Oversees how the captured pieces are displayed and how you can drop them
    def __init__(self,parent,pieces):
        
        self.width = 300
        self.height = 150
        self.master = parent
        tk.Canvas.__init__(self,master = parent, width = self.width, height = self.height, bg = "#FFE4B5",highlightthickness = 0)
        self.pieces = pieces
        self.captured = []
        self.draw_captured()
        
    def draw_captured(self):
        #Draws the captured pieces
        self.captured = []
        cpt = 0
        for i in self.pieces.values():
            if i.captured:
                column = cpt//12
                row = cpt%12
                i.draw(self,row*25,column*25)
                cpt += 1
                self.captured.append(i)
                
    def click(self,event):
        #Allows you to select a piece from the pieces you captured
        x = event.x
        y = event.y
        i = x//25 #Smaller pieces to allow all of them to appear on this board
        j = y//25
        indice = j*12 + i
        print(indice)
        print(self.captured)
        try: 
            #Does this if you clicked on a piece
            piece = self.captured.pop(indice)
            self.master.selectedPiece = piece
            
        except:
            #Does that if you clicked on an empty space
            self.master.selectedPiece = None
        
gameState = Game_state()
        
class Window(tk.Tk):
    #The main window of the game
    def __init__(self):
        
        tk.Tk.__init__(self)
        self.title("Shogi")

        
        self.configure(background='#FFEBCD')
        
        self.shownBoard = Board(self,gameState.board) #Displaying the board
        self.shownBoard.bind("<Button-1>",self.click)
        self.shownBoard.pack(side = "left",padx=30,pady=30)
        
        self.moveHistory = []
        self.chosenSide = ''
        
        self.whiteCaptured = Captured(self,gameState.es_pieces["b"]) 
        self.blackCaptured = Captured(self,gameState.es_pieces["w"])
        
        self.whiteCaptured.bind("<Button-1>",self.whiteCaptured.click)
        self.blackCaptured.bind("<Button-1>",self.blackCaptured.click)
        
        self.blackCaptured.pack(side = "top", padx = 10, pady = 10)
        self.whiteCaptured.pack(side = "bottom", padx = 10, pady = 10)
        
        self.promotion = "" #Used to know if you want/have to promote
        self.selectedPiece = None #Used to know whether you want to click to move or select a piece
    
        self.whichSide = SideWindow(self).disp()
        t = Timer(1,self.AIPlay) #Timer to allow a bit of time for the board to draw itself
        t.start()
    
    def AIPlay(self):
        if gameState.playing_side != self.chosenSide:
            print("Processing, side : ", gameState.playing_side)
            (_,movelist) = ABPruning.alphaBetaPruning(ABPruning.Node(gameState, gameState.playing_side, len(self.moveHistory), self.moveHistory),3)
            print("Done")
            move = movelist[0]
            self.moveHistory.append(move)
            gameState.update(move)
            
            
        else:
            print("Random time")
            move = ABPruning.randomMove(ABPruning.Node(gameState, gameState.playing_side, len(self.moveHistory), self.moveHistory))
            self.moveHistory.append(move)
            gameState.update(move)
            
        self.shownBoard.draw_pieces()
        self.blackCaptured.draw_captured()
        self.whiteCaptured.draw_captured()
        self.checkmate()
        t = Timer(1,self.AIPlay)
        t.start()
    
    def checkmate(self):
        if gameState.finished():
            if gameState.playing_side == "w":
                msg.showinfo("Victory !","BLACK WINS")
            else:
                msg.showinfo("Victory !","WHITE WINS")
    
    def click(self,event): 
        #What happens when you click somewhere on the board
        x = event.x
        y = event.y
        j = x//50
        i = y//50
        if self.selectedPiece == None: 
            #Either there is no selected piece, so you select one
            #This will also later display the moves you can do with the piece you've selected
            self.selectedPiece = gameState.board[i,j]
            self.shownBoard.highlight_reachable_squares(self.selectedPiece)
            
        else:
            #Or you've already selected a piece, and thus want to move it
            #For now, this allows illegal moves
            if self.selectedPiece.pos == None: 
                #Needed in case of a drop, a dropped piece had no prior positions
                oldi = None
                oldj = None
            else:
                
                (oldi,oldj) = self.selectedPiece.pos
                
            if oldi == i and oldj == j: 
                #If you want to unselect the piece you're holding
                self.selectedPiece = None
                self.shownBoard.highlight_reachable_squares(self.selectedPiece)
                
            else:
                s = gameState.playing_side
                print(self.selectedPiece)
                if s == self.selectedPiece.side: 
                    #If the right side is playing
                    if self.selectedPiece.pos == None:
                            #Case of a drop
                            prefix = self.selectedPiece.name[0]
                            self.selectedPiece = None #Unselecting
                            if prefix.upper() + "*" + id2str((i,j)) in gameState.legal_moves: 
                                #if you can drop there
                                drop = prefix.upper() + "*" + id2str((i,j))
                                gameState.update(drop) #Giving game_engine the needed info
                                self.moveHistory.append(drop)
                            else: 
                                #if you can't drop there
                                self.shownBoard.highlight_reachable_squares(None)
                                msg.showwarning("Illegal move","This drop is illegal")
                    
                    elif id2str(self.selectedPiece.pos) + id2str((i,j)) in gameState.legal_moves: 
                            #Else, if the move is valid
                            #Case of a move
                            if self.selectedPiece.must_promote((i,j)): #If you have to promote
                                
                                self.promotion = "+"
                                
                            elif self.selectedPiece.can_promote((i,j)): #If you can promote
                                
                                self.promotion_window = PromotionWindow(self).disp()
                                
                            else: #If you can't promote
                                
                                self.promotion = ""
                                
                            self.selectedPiece = None 
                            self.shownBoard.highlight_reachable_squares(None) 
                            move = id2str((oldi,oldj)) + id2str((i,j)) + self.promotion
                            gameState.update(move)
                            self.moveHistory.append(move)
                            self.promotion = "" #Resetting promotion variable
                            
                    else: 
                        #Wrong move
                        self.shownBoard.highlight_reachable_squares(None)
                        msg.showwarning("Illegal move","The selected piece is unable to do this move")
                        self.selectedPiece = None
                        
                else: 
                    #Wrong side
                    self.shownBoard.highlight_reachable_squares(None)
                    msg.showwarning("Illegal move","Wrong side")
                    self.selectedPiece = None
        print(id2str((i,j)))
        #Reload the pieces to draw them
        self.shownBoard.draw_pieces()
        self.blackCaptured.draw_captured()
        self.whiteCaptured.draw_captured()
        self.checkmate()
        print(gameState.legal_moves)
        t = Timer(1,self.AIPlay)
        t.start()
        
wind = Window()
wind.tk.mainloop()

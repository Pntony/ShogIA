
import shogi_engine
import copy
import time
import random
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup

piecesValue =  {'p' : 10,'l' : 43,'n' : 45,'s' : 64,'g' : 69,'b' : 89,'r' : 104, 'k' : 10000}
piecesValueC =  {'p' : 12,'l' : 48,'n' : 51,'s' : 72,'g' : 78,'b' : 111,'r' : 127}
piecesValueP =  {'p' : 42,'l' : 63,'n' : 64,'s' : 67,'b' : 115,'r' : 130}

def quickSortValue(l,low,high): #We use a quick sort to allow the alpha beta algorithm to prune even more branches
    if low < high:
            
            pivot = l[high].value
            i = low - 1
            for j in range(low,high):
                if l[j].value < pivot:
                    i += 1
                    l[i],l[j] = l[j],l[i]
            l[i+1],l[high] = l[high],l[i+1]
            
            index = i + 1
            quickSortValue(l, low, index - 1)
            quickSortValue(l, index + 1, high)
            
def cutDrops(moves):
    for i in moves:
        if '*' in i:
            if random.random() < 0.1:
                moves.remove(i)
    
class Node:
    
    def __init__(self,gs, s = 'b',d = 0, mvl = [], ):
        
        self.gameState = gs
        self.depth = d
        self.moveHistory = mvl
        self.sons = []
        self.value = 0
        self.aiSide = 'b'
        self.computeValue()
        
    def generate_sons(self):
        temp_gameState = shogi_engine.Game_state()
        for j in self.moveHistory:
                temp_gameState.update(j)
        self.moves = temp_gameState.legal_moves
        cutDrops(self.moves)
        
        for i in self.moves:
            copyTemp_gameState = copy.deepcopy(temp_gameState) #We have to make this weird roundabout instead of copying gameState because tkinter doesn't allow us to deepcopy anything it uses for some reason
            copyTemp_gameState.update(i)
            self.sons.append((Node(temp_gameState, shogi_engine.other_side(self.aiSide), self.depth + 1,self.moveHistory + [i])))
        quickSortValue(self.sons,0,len(self.sons) - 1)
        
        
    def computeValue(self): #Allows us to know whether this game state is good or bad for the AI
        #Divided in 3 parts

        #First part : counting pieces, simply counting how many pieces everyones owns
        pieceCount = 0
        for i in self.gameState.es_pieces['w']:
            piece = self.gameState.es_pieces['w'][i]
            if piece.captured:
                pieceCount-= piecesValueC[(piece.usi_name[-1])]
            elif piece.promotion == '+':
                pieceCount-= piecesValueP[(piece.usi_name[-1])]
            else:
                pieceCount-= piecesValue[(piece.usi_name[-1])]
        for i in self.gameState.es_pieces['b']:
            piece = self.gameState.es_pieces['b'][i]
            if piece.captured:
                pieceCount+= piecesValueC[(piece.usi_name[-1].lower())]
            elif piece.promotion == '+':
                pieceCount+= piecesValueP[(piece.usi_name[-1].lower())]
            else:
                pieceCount+= piecesValue[(piece.usi_name[-1].lower())]
                
        #Second part : is your king safe ? and how safe is the enemy's ? (divided in two sub parts)
        kingSafety = 0
        (bKingPos,wKingPos) = (self.gameState.get_king('b').pos,self.gameState.get_king('w').pos)
        
        #First sub part : How much freedom does your king have ? 
        #Checks how the king can move
        legal_moves = self.gameState.legal_moves
        usiBKingPos = shogi_engine.id2str(bKingPos)
        usiWKingPos = shogi_engine.id2str(wKingPos)
        
        wKingFreedom = 0
        bKingFreedom = 0
        for i in legal_moves:
            if i[:2] == usiBKingPos:
                bKingFreedom += 1
            elif i[:2] == usiWKingPos:
                wKingFreedom += 1
        
        kingFreedom = bKingFreedom - wKingFreedom
        kingSafety += 5*kingFreedom #multiplied by 5 becuase insignificant compared to other variables otherwise with our method
        
        
        #Second sub part : How many attackers and defenders are near your king ? And how strong are they ?
        #Being near the king = explained in a graphic in the files, different for attackers and defenders
        dDangerZone = [(2,-1),(2,0),(2,1),(1,-1),(1,0),(1,1),(0,-1),(0,1),(-1,-1),(-1,0),(-1,1)] #values for black king, multiply by -1 to get the one for the white king
        attackerWeight = [0,50,75,88,94,97,99] #values from Toga Log, varies with number of attackers
        #Thanks to this list, it is easier to know if we are facing a serious attack or just a lone piece wandering near the king
        kingDefenderValue = 0
        kingAttackerValue = 0
        attackerValue1,attackerCount1 = 0,0
        attackerValue2,attackerCount2 = 0,0
        
        for d in dDangerZone:
            xPosPiece1,yPosPiece1 = -d[0] + bKingPos[0],-d[1] + bKingPos[1]
            if 0 <= xPosPiece1 <= 8 and 0 <= yPosPiece1 <= 8:
                piece1 = self.gameState.board[xPosPiece1,yPosPiece1]
                if piece1 != None:
                    if piece1.side == 'w':
                        attackerValue1 += piecesValue[(piece1.usi_name[-1])] #The enemy's attackers make the value of your board go down
                        attackerCount1 += 1
                    else:
                        kingDefenderValue += piecesValue[(piece1.usi_name[-1].lower())] #Your defenders make the value of your board go up
            
            xPosPiece2,yPosPiece2 = d[0] + wKingPos[0],d[1] + wKingPos[1]
            if 0 <= xPosPiece2 <= 8 and 0 <= yPosPiece2 <= 8:
                piece2 = self.gameState.board[xPosPiece2,yPosPiece2] 
                if piece2 != None:
                    if piece2.side == 'b':
                        attackerValue2 += piecesValue[(piece2.usi_name[-1].lower())]
                        attackerCount2 += 1
                    else:
                        kingDefenderValue += piecesValue[(piece2.usi_name[-1])] 
                    
        kingAttackerValue = - attackerWeight[min(attackerCount1,6)]/100*attackerValue1 + attackerWeight[min(attackerCount2,6)]/100*attackerValue2

        kingSafety += kingAttackerValue + kingDefenderValue
        
        #Third part : gauging how much of the board is under your control
        #To gauge board control, you look at a square and see which pieces can reach it.
        #Then you add the piece's value to a total to determine who has the most control over the board
        
        PRI = self.gameState.PRI #Loading PRI
        boardControl = 0
        
        for i in range(9):
            for j in range(9): #We go through the whole board
                if piece.name.lower() != 'k':
                    for piece in PRI['b'][i,j]:
                        boardControl += piecesValue[(piece.usi_name[-1].lower())]
                        
                    for piece in PRI['w'][i,j]:
                        boardControl -= piecesValue[(piece.usi_name[-1])]
        
        self.value = kingSafety/500 + pieceCount + boardControl/100
        #print('Total value : ',self.value)
        #Invert value according to which side the AI is playing on
        if self.aiSide != 'b':
            self.value *= -1
        
    def __str__(self):
        return 'Depth : {} \nMove history : {} \nSons : {} \nValue : {}'.format(self.depth,self.moveHistory, self.sons, self.value)

#for the opening book, we need to establish a relation with the website containing it
options = Options()
options.headless = True
driver = webdriver.Firefox(options = options)
url = 'https://www.crazy-sensei.com/book/shogi'
        
def randomMove(Node):
    return random.choice(Node.gameState.legal_moves)        

#In the following function, we replace -inf and +inf by 10000, because the values 
#of game states will never go above those thresholds
global book
    
def alphaBetaPruning(Node, depth, alpha = -100000, beta = 100000, maximizingPlayer = True,book = True):
    
    if depth == 0 or Node.gameState.finished():
        return (Node.value,[])
    
    if Node.depth < 12 and book: #Opening book
        print('Trying to use opening book')
        if Node.moveHistory != []:
            appendix = '/' + ','.join(Node.moveHistory)
            driver.get(url + appendix)
        else: 
            print('Opening move')
            driver.get(url)
            
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')
        
        negamax = soup.findAll('p')[1].text[9:]
        print('Negamax :',negamax)
        if negamax[:4] != 'tion':
            print('Current position is in the book')
            movelist = []
            for a in soup.findAll('th'):
                movelist.append(a.text)
            movelist = movelist[3:]
            moveNumber = 0
            if random.random() < 0.8:
                moveNumber += 1
            print('Move found : ',movelist[moveNumber%len(movelist)])
            return (negamax,[movelist[moveNumber%len(movelist)]])
    
    
    Node.generate_sons()

    if maximizingPlayer:
        v = -100000
        for son in Node.sons:
            (v2,mvl) = alphaBetaPruning(son, depth - 1, alpha, beta, False, False)
            if v < v2:
                v = v2
                move = son.moveHistory[-1]
                movelist = mvl
            alpha = max(alpha,v)
            if alpha >= beta:
                break
        return (v,[move] + movelist)
        
    else:
        v = 100000
        for son in Node.sons:
            (v3,mvl) = alphaBetaPruning(son, depth - 1, alpha, beta, True, False)
            
            if v3 < v:
                v = v3
                move = son.moveHistory[-1]
                movelist = mvl
            beta = min(beta,v)
            if alpha >= beta:
                break
        return (v,[move] + movelist)
        

#state0 = Node(shogi_engine.Game_state())
#(gain,movelist) = alphaBetaPruning(state0,3)
#driver.quit

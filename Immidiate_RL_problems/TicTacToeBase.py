import numpy as np
import os
os.system('cls' if os.name == 'nt' else 'clear')

gameBoard = np.chararray((3,3), itemsize = 1);
gameBoard[:] = '';
N = 3;
inputs = {"q" : (0,0), "w" : (0,1), "e" : (0,2), 
		  "a" : (1,0), "s" : (1,1), "d" : (1,2), 
		  "z" : (2,0), "x" : (2,1), "c" : (2,2) };


winningPatterns = [{0 : 0, N + 1 : 0},{0 : 0, N + 1 : 0}];

markings = {0 : "O", 1 : "X"};
player = 0;
count = 0;

while(count < 9):

	print (gameBoard);
	print("Player" + str(player + 1) + ": your turn"); 
	ip = raw_input();
	os.system('cls' if os.name == 'nt' else 'clear')
	
	if ip not in inputs:
		print("You have pressed the wrong key. Try again");
		continue;
	
	if gameBoard[inputs[ip]].isalpha():
		print("The chosen box has already been marked. Try again");
		continue;
	
	gameBoard[inputs[ip]] = markings[player];
	
	
	if inputs[ip][0] + 1 not in winningPatterns[player]:
		winningPatterns[player][inputs[ip][0] + 1] = 0;	

	if -(inputs[ip][1] + 1) not in winningPatterns[player]:
		winningPatterns[player][-inputs[ip][1] - 1] = 0;	


	winningPatterns[player][inputs[ip][0] + 1] += 1;
	winningPatterns[player][-inputs[ip][1] - 1] += 1;

	if inputs[ip][0] == inputs[ip][1]:
		winningPatterns[player][0] += 1;


	if inputs[ip][0] + inputs[ip][1] == N - 1:
		winningPatterns[player][N + 1] += 1;
        


	if winningPatterns[player][-inputs[ip][1] - 1] == N or winningPatterns[player][inputs[ip][0] + 1] == N or winningPatterns[player][0] == N or winningPatterns[player][N + 1] == N: 
		print("Player" + str(player + 1) + " wins. Congratulations!!!");
		break;


	player = (player + 1) % 2;
	count += 1;

if count == 9:
	print("Game Drawn");
print (gameBoard);

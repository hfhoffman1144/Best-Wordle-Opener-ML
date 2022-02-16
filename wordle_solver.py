import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from typing import List
from enum import Enum
from collections import Counter
from tqdm import tqdm
from pyod.models.copod import COPOD

# Create a list of each letter in the alphabet
ALPHABET = list(string.ascii_lowercase)

def reduce_word_bank(guess:str, guess_pattern:List[str], word_bank:List[str]) -> List[str]:
    
    # Make sure user has entered a valid pattern
    for color in guess_pattern:
        
        if (color not in {'green','grey','yellow'}):
            
            raise(Execption(f'Invalid pattern: value {color} not understood'))
            
    guess_array:np.ndarray = np.array(list(guess))
    
    # Convert user pattern to enum
    enum_pattern:np.ndarray = np.array([Pattern[u].value for u in guess_pattern])
        
    new_word_bank:List[str] = word_bank.copy()
    
    # Iterate through each position in the pattern
    for pos in range(len(enum_pattern)):
        
        # Letter corresponding to the position
        curr_letter:str = guess_array[pos]
        
        # Check grey condition - the value is grey and the letter only appears one time in the word
        if (enum_pattern[pos] == Pattern.grey.value) and (np.sum(guess_array==curr_letter)==1):

            new_word_bank:List[str] = [w for w in new_word_bank if curr_letter not in w]

        # Check yellow condition
        if enum_pattern[pos] == Pattern.yellow.value:

            # Determine the least number of possible occurences of curr_letter in the word
            min_occur:int = np.sum(enum_pattern[np.where(guess_array==curr_letter)] == Pattern.yellow.value)

            new_word_bank:List[str] = [w for w in new_word_bank if np.sum(np.array(list(w))==curr_letter)\
                                       >= min_occur]

        # Check green condition
        if enum_pattern[pos] == Pattern.green.value:
            
            new_word_bank:List[str] = [w for w in new_word_bank if w[pos] == curr_letter]
                
    if guess in new_word_bank:
        new_word_bank.remove(guess)
        
    return new_word_bank

class Pattern(Enum):
    '''
    Enum representation of Wordle feedback
    '''
    grey = 0
    yellow = 1
    green = 2
    
class WordleGuess:
    
    '''
    Make a Wordle guess from a given word bank
    '''
    
    def __init__(self, word_bank:List[str]):
        
        if len(word_bank) <= 1:
            
            raise(Exception('Not enough words in the word bank'))
        
        self.word_bank = word_bank
        self.word_encoding_cols = [f'{letter}{position}' for letter in ALPHABET for position in range(5)]
        
    def _create_word_encoding(self) -> pd.DataFrame:
        
        # Create letter/position occurrence matrix
        word_encoding_data = pd.DataFrame()

        # For each word in the word bank
        for word in self.word_bank:

            # Convert the word to it letter-position format
            word_and_pos = ''.join([f"{letter}{pos}" for pos, letter in enumerate(word)])

            # Create letter-position counter dictionary
            letter_pos_counter = {}
            for wp in self.word_encoding_cols:
                
                letter_pos_counter[wp] = len(re.findall(wp, word_and_pos))

            tmp_encoding_data = pd.DataFrame(letter_pos_counter, index=[word])
            word_encoding_data = pd.concat([word_encoding_data, tmp_encoding_data])
            
        # Drop columns with all zeros
        for col in word_encoding_data.columns:

            if word_encoding_data[col].sum() == 0:
                word_encoding_data.drop(col, axis=1, inplace=True)
                
        return word_encoding_data
    
    def make_guess(self) -> str:
        
        # Create word encoding data
        word_encoding_data = self._create_word_encoding()
        
        # Fit COPOD model
        copod_model = COPOD(contamination=0.01)
        copod_model.fit(word_encoding_data)
        
        word_encoding_data['score'] = copod_model.decision_scores_
        word_encoding_data.sort_values('score',inplace=True)
        
        return word_encoding_data.index[0]
    
class WordleSimulation:
    
    '''
    Simulate a Wordle game where the COPOD algorithm generates guesses
    '''
    
    def __init__(self, target:str, word_bank:List[str], first_guess:str = None):
        
        self.target = target
        self.word_bank = word_bank
        self.first_guess = first_guess
        
    def _get_pattern_list(self, guess:str) -> List[str]:
    
        # Convert strings to numpy arrays
        guess_array:np.ndarray = np.array(list(guess))
        target_array:np.ndarray = np.array(list(self.target))

        if len(guess) != len(self.target):

            raise(Exception(f'Cannot compare words with different lengths'))

        pattern:List[str] = []

        for pos in range(len(guess)):

            # Check grey condition
            if guess_array[pos] not in target_array:

                pattern.append('grey')

            # Check green condition
            if guess_array[pos] == target_array[pos]:

                pattern.append('green')

            # Check yellow condition
            if (guess_array[pos] in target_array) and (guess_array[pos] != target_array[pos]):

                # The number of times the current letter appears in the word
                num_letter_occur:int = np.sum(np.array(target_array) == guess[pos])

                if num_letter_occur <= np.sum(guess_array[0:len(pattern)] == guess_array[pos]):

                    pattern.append('grey')

                else:

                    pattern.append('yellow')

        return pattern

    def run_simulation(self) -> int:
        
        new_word_bank:List[str] = self.word_bank.copy()
        
        # Make initial guess
        if self.first_guess is not None:
            
            guess:str = self.first_guess
                
        else:
            
            guess:str = WordleGuess(new_word_bank).make_guess()
                
        if guess == self.target:
            
            return 1
            
        # Number of guesses
        num_guesses:int = 0
        
        while guess != self.target:
            
            num_guesses +=1
            
            # Get pattern for the guess
            pattern:List[str] = self._get_pattern_list(guess)
            
            # Update word bank based on pattern
            new_word_bank:List[str] = reduce_word_bank(guess, pattern, new_word_bank)
            
            if len(new_word_bank) == 1:
                
                return num_guesses+1
            
            guess:str = WordleGuess(new_word_bank).make_guess()
        
        if num_guesses == 1:
            num_guesses+=1
            
        return num_guesses
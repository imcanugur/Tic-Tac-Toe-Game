'''

    Copyright © 2022 | Uğur CAN
    www.ugurcan.rf.gd

'''

import numpy as np
import random
import pickle

# Oyun Bloğu Başlangıç
class XOX:
    def __init__(self):
        self.CurrentState = np.zeros(9, dtype = np.int8)
        self.winner = None
        self.player = 1
        
    def DrawCurrentGame(self):
        CurrentState = ['X' if x == 1 else 'O' if x == -1 else '---' for x in self.CurrentState]
        print(f'{CurrentState[0]:^5} {CurrentState[1]:^5} {CurrentState[2]:^5}')
        print(f'{CurrentState[3]:^5} {CurrentState[4]:^5} {CurrentState[5]:^5}')
        print(f'{CurrentState[6]:^5} {CurrentState[7]:^5} {CurrentState[8]:^5}')
        print('*'*40)
        
    def GetCurrentGame(self):
        return self.CurrentState
    
    def GetCurrentGameTuple(self):
        return tuple(self.CurrentState)

    def get_available_positions(self):
        return(np.argwhere(self.CurrentState==0).ravel())

    def reset_game(self):
        self.CurrentState = np.zeros(9, dtype = np.int8)
        self.player = 1

    def get_player(self):
        return self.player

    def make_move(self, action):
            if action in self.get_available_positions():
                self.CurrentState[action] = self.player
                self.player *= -1
            else:
                print('Zaten Dolu Olan Bir Bloğa Giriş Yaptınız')
                print('*'*40)

    def MakeMove(self, _CurrentState, action):
        _CurrentState[action] = self.player
        return _CurrentState

    def get_next_states(self):
        states = []
        _CurrentState = self.CurrentState
        _available_moves = self.get_available_positions()
        for move in _available_moves:
            states.append(self.MakeMove(_CurrentState = _CurrentState, action=move))
        return states

    def is_winner(self, isgame = False):
        winner_coordinates = np.array([[0,1,2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])
        for coordinate in winner_coordinates:
            total = sum(self.CurrentState[coordinate])
            if total == 3:
                if isgame:
                    print('Beni Yendin.') # kullanıcı kazandı
                    print('*'*40)
                    print('Bu Sefer Şanlıydın Bir Daha Oynayalım')
                    print('*'*40)
                self.winner = 1
                self.reset_game()
                return 1
            elif total == -3:
                if isgame:
                    print("Ben Kazandım:)") # Yapay zeka kazandı
                    print('*'*40)
                    print('Tekrar Oynayalım İstersen')
                    print('*'*40)
                self.winner = -1
                self.reset_game()
                return -1
            
            elif sum(self.CurrentState == 1) == 5:
                if isgame:
                    print('Kazanan Çıkmadı.')
                    print('*'*40)
                    print('Bir Daha Oynayıp Görelim Kim Daha İyi')
                    print('*'*40)
                self.winner = -2
                self.reset_game()
                return -2
        return False
# Oyun Bloğu Sonu

# Yapay Zeka Öğrenmesi Bloğu Başlangıç
class Agent:
    def __init__(self, game, player = 'User', episode = 100000, epsilon = 0.9, discount_factor = 0.6, eps_reduce_factor = 0.01):
        self.game = game
        self.player = player # Yapay Zeka
        self.brain = dict() # Oyundaki Diğer Değerlerin Tutulması
        self.episode = episode # Yapay Zekanın Ne Kadar Tekrar Edeceğini Alır 
        self.epsilon = epsilon # Bu Aşamada Üretilen Rastgele Sayıların Veya Daha Önce Kayıdı Yapılan Sayıların Ne Sıklıkla Kullanılacağını Söyler
        self.discount_factor = discount_factor # Rastgele Veya Daha Önce Üretilen Sayıların Kullanılma Katsayısının Azalıp Ulaşacağı Minimum Katsayı
        self.results = {'User' : 0, 'Machine': 0, 'Draw': 0}
        self.eps_reduce_factor = eps_reduce_factor # Rastgele Veya Daha Önce Üretilen Sayıların Kullanılma Katsayısının Azalma Oranı

    def IntelligenceLearningSave(self, player):
        with open('BrainMachine', 'wb') as BrainFile:
            pickle.dump(self.brain, BrainFile)

    def IntelligenceLearningLoad(self, player):
        try:
            with open('BrainMachine', 'rb') as BrainFile:
                self.brain = pickle.load(BrainFile)
        except:
            print('Makine öğrenmesi bulunmadı. Rastgele hamlelerle oynanacak.') 
            print('*'*40)

    def reward(self, player, move_history, result):
        _reward = 0
        if player == -1:
            if result == 1:
                _reward = -1
                self.results['User'] += 1 
            elif result == -1:
                _reward = 1
                self.results['Machine'] += 1
        if result == -2:
             self.results['Draw'] += 1
        move_history.reverse()
        for state, action in move_history:
            self.brain[state, action] = self.brain.get((state, action), 0.0) + _reward
            _reward *= self.discount_factor
            
    def UseBrain(self):
        possible_actions = self.game.get_available_positions()
        max_qvalue = -1000
        best_action = possible_actions[0]
        for action in possible_actions:
            qvalue = self.brain.get((self.game.GetCurrentGameTuple(), action), 0.0)
            if qvalue > max_qvalue:
                best_action = action
                max_qvalue = qvalue
            elif qvalue == max_qvalue and random.random() < 0.5:
                best_action = action
                max_qvalue = qvalue
            elif len(possible_actions) == 9:
                best_action = random.choice(possible_actions)
                break
        return best_action
    
    def IntelligenceLearning(self):
        try:
            for _ in range(self.episode):
                if _ % 1000 == 0:
                    print('Tamamlanan Similasyon: '+str(_))
                    self.epsilon -= self.eps_reduce_factor
                move_history = []
                while True:
                    available_actions = self.game.get_available_positions()
                    action_x = random.choice(available_actions)
                    self.game.make_move(action_x)
                    if self.game.is_winner():
                        self.reward(-1 ,move_history, self.game.winner)
                        break
                    if random.random()<self.epsilon:
                        available_actions = self.game.get_available_positions()
                        ArtificialIntelligence = random.choice(available_actions)
                        move_history.append([self.game.GetCurrentGameTuple(), ArtificialIntelligence])
                        self.game.make_move(ArtificialIntelligence)

                    else:
                        ArtificialIntelligence = self.UseBrain()
                        move_history.append([self.game.GetCurrentGameTuple(), ArtificialIntelligence])
                        self.game.make_move(ArtificialIntelligence)
                        break

                    if self.game.is_winner():
                        self.reward(-1 ,move_history, self.game.winner)
                        break

            self.IntelligenceLearningSave('BrainMachine')
            print('*'*40)
            print('Yapay Zeka Öğrenmesi Tamamlandı. ('+str(OgrenmeSayisi)+')')
            print('*'*40)
            '''
            Similasyondaki Sonuçlar
            print('Öğrenim Sonuçları: ') 
            print(self.results)
            '''
        except:
            print('*'*40)
            print("Yapay Zeka Öğrenmesi Tamamlanamadı. Tekrar Deneyin.")
            print('*'*40)
        
    def Play(self): 
        self.IntelligenceLearningLoad(self.player)
        order = 1 if self.player=='X' else -1
        while True:
            if order == 1:
                print("Yapay Zekanın Hamlesi")
                print('*'*40)
                self.game.make_move(self.UseBrain())
                self.game.DrawCurrentGame()
                order *= -1
                if self.game.is_winner(isgame = True):
                    break
            else:
                try:
                    ArtificialIntelligence = int(input('Hangi Kare: '))
                    print('*'*40)
                    if ArtificialIntelligence < 10 and ArtificialIntelligence > 0:
                        try:
                            print('Sizin Hamleniz')
                            print('*'*40)
                            self.game.make_move(ArtificialIntelligence-1)
                            self.game.DrawCurrentGame()
                            order *= -1
                            if self.game.is_winner(isgame = True):
                                break
                        except:
                            print('*'*40)
                            print("Girdi Bulunamadı. Tekrar Deneyin.")
                            print('*'*40)
                    else: 
                        print("Yanlış Giriş Lütfen 1 ve 9 Arasında Bir Sayı Girin")
                        print('*'*40)
                        
                except:
                    print('*'*40)
                    print("Girdi Bulunamadı. Tekrar Deneyin.")
                    print('*'*40)
                

# Yapay Zeka Öğrenmesi Bloğu Son
print('*'*40)
print(' '*15+'Uğur CAN')
print('*'*40)
print(' '*17+'XOX')
print('*'*40)

while True:
    try:
        isim = input('Merhaba Adın Ne: ')
        print('*'*40)
        print('Oyunuma Hoşgeldin',isim)
        print('*'*40)
        break
    except :
        print('\n'+'*'*40)
        print("Girdi Bulunamadı. Tekrar Deneyin.")
        print('*'*40)



while True:
    secim = input("0: Yapay Zeka Öğrenmesi\n1: Oyuna gir\n2: Nasıl Oynanır\n3: Çıkış\nNe Yapmak İstersin: ")
    print('*'*40)
    game = XOX()
    OgrenmeSayisi = 0
    if secim == '0': # Yapay Zeka Öğrenmesi İçin Veri Alma
        try:
            OgrenmeSayisi = input("Makine Öğrenme Sayısını Girin:")
            print('*'*40)
        except:
            print('*'*40)
            print("Girdi Bulunamadı. Tekrar Deneyin.")
            print('*'*40)
        
    agent = Agent(game, 'Machine',discount_factor = 0.6, episode = int(OgrenmeSayisi)) # Yapay Zeka Öğrenmesi İçin Alınan Veriyi Atama

    if secim == '0': # Yapay zeka öğrenmesi
        agent.IntelligenceLearning()

    elif secim == '1': # Oyun Başlangıçı
        agent.Play()
    elif secim == '2':
        print('Merhaba ' + isim + " Oyunumu Anlatim.\nX-> Sensin\nO-> Yapay Zeka\nSağdan Sola ve Yukarıdan Aşağı Olacak Şekilde 1'den 9'a Kadar Bloklar Numaralandırıldı. Şu Şekilde:\n-1- -2- -3-\n-4- -5- -6-\n-7- -8- -9-\nVe Senden Bu Bloklardan Birini Seçmen Bekleniyor.\nİlk Oynamada Yapay Zeka Eğiticisini Yapmayı Unutma Yoksa Kolay Mod Olup Herhangi Bir Algoritma Kullanmadan Rastgele Oynar.\nVe Mümkün Olduğunca Yapay Zekanın Yüksek Miktarda Simüle Etmesine Özen Göster.\nO Zaman Gerçek Bir Kişi İle Oynadığını Düşüneceksin.")
        print('*'*40)
    elif secim == '3':
        break
    else:
        print('Lütfen doğru bir seçim yapın.')
        print('*'*40)
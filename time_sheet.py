from datetime import datetime, timedelta
import numpy as np

# t1 = datetime(2060, 12, 11)
# t2 = datetime(2060, 12, 20)

# print(t1-t2)

time = [58.842082023620605, 58.96583437919617, 58.82274580001831, 59.20848059654236, 59.11894178390503, 59.18281698226929, 59.36720252037048, 59.62696051597595, 59.31802201271057, 60.03097414970398, 59.478829860687256, 59.533921241760254, 59.410242319107056, 59.565486431121826, 59.66751217842102, 59.17000985145569, 58.600992918014526, 58.68800640106201, 58.660558462142944, 58.96265983581543, 58.73112463951111, 58.8916175365448, 58.7081401348114, 58.603437185287476, 58.56492352485657, 58.54685831069946, 58.44475531578064, 58.76238465309143, 58.70250940322876, 58.557493448257446]

print(np.mean(time), np.std(time)/np.sqrt(30))
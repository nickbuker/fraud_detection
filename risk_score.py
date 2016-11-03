import numpy as np

def risk_score(proba, gross_sales):
   """
   Input: proba and gross_sales can be np arrays, pd series,
   floats, or ints of equal length
   Return: risk score (float)
   """
   log_gs = np.log10(gross_sales + 1)
   return proba * log_gs

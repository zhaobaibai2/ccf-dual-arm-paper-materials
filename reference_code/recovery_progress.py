import numpy as np
from .progress_control import MeasuredProgress
class RecoveryProgress(MeasuredProgress):
 def update(self,state):
  before=self.index;p=super().update(state)
  if self.index!=before:return p
  # Recover a modest look-ahead overshoot only within unchanged hand posture.
  end=min(len(self.q),before+26);states=self.q[before:end]
  hand=self.mask.copy();hand[:14]=False
  if np.any(np.max(abs(states-self.q[before])[:,hand],axis=0)>2.):return p
  errors=np.max(abs(states-state)[:,self.mask]/self.radius[self.mask],axis=1)
  nearest=int(errors.argmin())
  if errors[nearest]<=1 and nearest>0:self.index+=min(nearest,4)
  return self.index/(len(self.q)-1)

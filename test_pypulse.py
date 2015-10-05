#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pypulse


print('Creating pupulse instance')
model = pypulse.PyPulse()

model.clear()
model.append(action=1, observation=1, reward=.1)
model.append(action=2, observation=1, reward=.1)

# this takes quite long!
# model.fit()

print('DONE')

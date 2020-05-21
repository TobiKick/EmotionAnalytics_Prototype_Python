#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pygetwindow as gw
allTitles = gw.getAllTitles()

visibleWindows = []
for i in allTitles:
    notepadWindow = gw.getWindowsWithTitle(i)[0]
    print(notepadWindow.title)
    if notepadWindow.isMinimized or notepadWindow.isMaximized:
        print(notepadWindow.title)
        visibleWindows.append(notepadWindow.title)


# In[ ]:





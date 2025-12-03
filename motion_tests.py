# Slow straight (+x)
await set_motors(node, 180, 180);   await asyncio.sleep(1.2)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Left arc (~+x to +y), gentle curvature
await set_motors(node, 240, 280);   await asyncio.sleep(1.0)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Medium straight (+y)
await set_motors(node, 260, 260);   await asyncio.sleep(1.2)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Right arc (~+y back to +x), mirror of the left arc
await set_motors(node, 280, 240);   await asyncio.sleep(1.0)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Fast straight (+x)
await set_motors(node, 320, 320);   await asyncio.sleep(1.0)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Left arc (~+x to +y), same curvature as before
await set_motors(node, 240, 280);   await asyncio.sleep(1.0)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Slow straight (+y)
await set_motors(node, 180, 180);   await asyncio.sleep(1.2)
await set_motors(node, 0, 0);       await asyncio.sleep(0.6)

# Right arc (~+y back to +x), mirror
await set_motors(node, 280, 240);   await asyncio.sleep(1.0)
await set_motors(node, 0, 0);       await asyncio.sleep(0.8)



TEST 2: 

# Cycle 1 (x then y)

# Slow straight (+x)
await set_motors(node, 180, 180);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Left arc (+x -> +y), gentle curvature
await set_motors(node, 240, 280);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Medium straight (+y)
await set_motors(node, 260, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Right arc (+y -> +x), mirror of left arc
await set_motors(node, 280, 240);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Fast straight (+x)
await set_motors(node, 320, 320);   await asyncio.sleep(0.80)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Cycle 2 (y then x), symmetric order

# Left arc (+x -> +y)
await set_motors(node, 240, 280);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Slow straight (+y)
await set_motors(node, 180, 180);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Right arc (+y -> +x)
await set_motors(node, 280, 240);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Medium straight (+x)
await set_motors(node, 260, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Left arc (+x -> +y)
await set_motors(node, 240, 280);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# Fast straight (+y)
await set_motors(node, 320, 320);   await asyncio.sleep(0.80)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

TEST 3:

# Pause initiale (mesures statiques)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 1) Droit en +x
await set_motors(node, 260, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 2) Arc doux gauche (~+x -> +y), même courbure utilisée plus tard en miroir
await set_motors(node, 260, 300);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 3) Droit en +y
await set_motors(node, 260, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 4) Arc doux droit (~+y -> +x), miroir de l’étape 2
await set_motors(node, 300, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 5) Droit en +x (répète le test x)
await set_motors(node, 260, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 6) Arc doux gauche (~+x -> +y), même paramètres pour la symétrie
await set_motors(node, 260, 300);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 7) Droit en +y (répète le test y)
await set_motors(node, 260, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

# 8) Arc doux droit (~+y -> +x) pour revenir à l’orientation initiale
await set_motors(node, 300, 260);   await asyncio.sleep(1.00)
await set_motors(node, 0, 0);       await asyncio.sleep(0.30)

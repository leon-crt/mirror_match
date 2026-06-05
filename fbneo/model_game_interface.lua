package.path = "lua_libs/luasocket/?.lua;"
package.cpath = "lua_libs/socket/core.dll;" .. "lua_libs/mime/core.dll;"
local socket = require('socket')

Frame_counter = 1
Buff_size = 100
StunnedP1, StunnedP2 = false, false
CanRecoverFromStunP1, CanRecoverFromStunP2 = false, false
HitStateP1, HitStateP2 = nil, nil
RoundNumber = 0
Host, Port = "127.0.0.1", 42069
Timeout = 0.0015
Desynced = false

function Split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t = {}
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    table.insert(t, str)
  end
  return t
end

-- hitstun detection function
function IsHit(player, hitState, hit, health, state)
    if (health < player.previousHealth)
    then
        hit = 1
        hitState = state
    end

    if hitState ~= nil
    then
        if hitState ~= state -- if the state has changed since the character got hit we can assume they're not in hitstun anymore
        then
            hit = 0
            hitState = nil
        else
            hit = 1
        end
    end
    
    return hit, hitState
end

function StunHandler(player, stun, stunned, state, isStunned, canRecoverFromStun)
    if (stun == 0 and player.previousStun > 10)
    then
        stunned = true
    end
    
    if stunned
    then
        if state == 70 -- state 70 doesnt always mean stunned, but when stunned state should always be 70 at some point I think??
        then
            canRecoverFromStun = true
        end
        -- First condition means that at one point the character was stunned and now it's not anymore. second means the character was hit while stunned which causes them to not be stunned anymore
        if ((state ~= 70 and canRecoverFromStun) or (stun > 0))
        then
            stunned = false
            canRecoverFromStun = false
            isStunned = 0
        else
            isStunned = 1
        end
    end
    return stunned, canRecoverFromStun, isStunned
end

function FormatState(p1, p2)
    local p1_data = string.format("%d,%d,%d,%d,%d,%d,%d,%d", p1.posX[#p1.posX], p1.posY[#p1.posY], p1.health[#p1.health], p1.super[#p1.super], p1.stun[#p1.stun], p1.isStunned[#p1.isStunned], p1.hit[#p1.hit], p1.thrown[#p1.thrown])
    local p2_data = string.format("%d,%d,%d,%d,%d,%d,%d,%d", p2.posX[#p2.posX], p2.posY[#p2.posY], p2.health[#p2.health], p2.super[#p2.super], p2.stun[#p2.stun], p2.isStunned[#p2.isStunned], p2.hit[#p2.hit], p2.thrown[#p2.thrown])
    local p2_inp = p2.inputs[#p2.inputs]
    local p2_inputs = string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", p2_inp["Left"] or 0, p2_inp["Up"] or 0, p2_inp["Right"] or 0, p2_inp["Down"] or 0, p2_inp["Weak Punch"] or 0, p2_inp["Medium Punch"] or 0, p2_inp["Strong Punch"] or 0, p2_inp["Weak Kick"] or 0, p2_inp["Medium Kick"] or 0, p2_inp["Strong Kick"] or 0, p2_inp["Start"] or 0, p2_inp["Coin"] or 0)
    
    -- add padding to make every message the same length
    local raw_string = p1_data .. ',' .. p2_data .. ',' .. p2_inputs .. ','
    local padded_string = ''
    local padding_len = 99 - #raw_string
    for _=0,padding_len do
        padded_string = padded_string .. '#'
    end
    padded_string = padded_string .. raw_string
    return padded_string
end

BToN = { [true] = 1, [false] = 0}
SToB = { ['0'] = false, ['1'] = true}

StateData = 
{
    characterId = "",
    superId = "",
    posX = {},
    posY = {},
    health = {},
    super = {},
    stun = {},
    hit = {},
    inputs = {}
}

function StateData:new (chid, suid, posx, posy, health, previousHealth, super, maxSuperBar, stun, previousStun, isStunned, hit, thrown, inputs)
    local o = {}
    setmetatable(o, {__index = self})
    o.characterId = chid or ""
    o.superId = suid or ""
    o.superBarLength = maxSuperBar or 0
    o.previousStun = previousStun or 0
    o.previousHealth = previousHealth or 0
    o.previousInput = Split('0,0,0,0,0,0,0,0,0,0,0,0', ',')
    o.posX = posx or {}
    o.posY = posy or {}
    o.health = health or {}
    o.super = super or {}
    o.stun = stun or {}
    o.isStunned = isStunned or {}
    o.thrown = thrown or {}
    o.hit = hit or {}
    o.inputs = inputs or {}
    return o
end

function StateData:wipe()
    self.previousStun = self.stun[#self.stun]
    self.previousHealth = self.health[#self.health]
    self.previousInput = self.inputs[#self.inputs]
    self.posX = {}
    self.posY = {}
    self.health = {}
    self.super = {}
    self.stun = {}
    self.isStunned = {}
    self.hit = {}
    self.thrown = {}
    self.inputs = {}
end

function StateData:update(posx, posy, health, super, stun, isStunned, hit, thrown, inputs)
    self.previousStun = stun
    self.previousHealth = health
    self.previousInput = self.inputs[#self.inputs] or  Split('0,0,0,0,0,0,0,0,0,0,0,0', ',')
    table.insert(self.posX, posx)
    table.insert(self.posY, posy)
    table.insert(self.health, health)
    table.insert(self.super, super)
    table.insert(self.stun, stun)
    table.insert(self.isStunned, isStunned)
    table.insert(self.hit, hit)
    table.insert(self.thrown, thrown)
    table.insert(self.inputs, inputs)
end

-- Initialize buffer classes
P1 = StateData:new()
P2 = StateData:new()

function GameInterface()
    -- P1 and P2 state values
    local posXP1, posYP1, posXP2, posYP2
    local healthP1, healthP2
    local superP1, superP2
    local superCountP1, superCountP2
    local stunP1, stunP2
    local isStunnedP1, isStunnedP2 = 0, 0
    local hitP1, hitP2 = 0, 0
    local beingThrownP1, beingThrownP2
    local stateP1, stateP2

    -- Get current game phase
    local in_match = memory.readbyte(0x020154A7)  -- 1 = match intro, 2 = after round start, 9 = character select, 6 = end of round, 8 = transition between rounds

    if in_match == 9
    then
        -- for now do nothing but maybe could auto select the characters and speed up to round start
    elseif in_match == 2 -- after round start
    then
        -- Extract P1 and P2 state values
        posXP1, posYP1 = memory.readwordsigned(0x02068CD0), memory.readwordsigned(0x02068CD4)
        posXP2, posYP2 = memory.readwordsigned(0x02069168), memory.readwordsigned(0x0206916C) 
        -- Summing 1 to health value because we want 0 health to mean dead instead of still alive with one pixel left
        healthP1, healthP2 = memory.readbyte(0x02028655) + 1, memory.readbyte(0x0202866D) + 1
        superP1, superP2 = memory.readbyte(0x020286A5), memory.readbyte(0x020286D9) -- saBarContent1 -> 0x020286A5   saBarCount1 -> 0x020286AB    bar1 -> 0x020695B5
        superCountP1, superCountP2 = memory.readbyte(0x020695BF), memory.readbyte(0x020695EB)
        superP1 = superP1 + superCountP1 * P1.superBarLength
        superP2 = superP2 + superCountP2 * P2.superBarLength
        -- Throws
        beingThrownP1, beingThrownP2 = BToN[memory.readbyte(0x02068C6C + 0x3CF) ~= 0], BToN[memory.readbyte(0x02069104 + 0x3CF) ~= 0]
        -- Stun management hell
        stunP1 = bit.rshift(memory.readdword(0x020695F7 + 0x6), 24) -- stun -> 0x02028805  stunstatus -> 0x020695FD
        stunP2 = memory.readbyte(0x02028829)
        stateP1, stateP2 = memory.readbyte(0x02068E75), memory.readbyte(0x020691B3)

        StunnedP1, CanRecoverFromStunP1, isStunnedP1 = StunHandler(P1, stunP1, StunnedP1, stateP1, isStunnedP1, CanRecoverFromStunP1)
        StunnedP2, CanRecoverFromStunP2, isStunnedP2 = StunHandler(P2, stunP2, StunnedP2, stateP2, isStunnedP2, CanRecoverFromStunP2)

        -- hitstun detection
        hitP1, HitStateP1 = IsHit(P1, HitStateP1, hitP1, healthP1, stateP1)
        hitP2, HitStateP2 = IsHit(P2, HitStateP2, hitP2, healthP2, stateP2)

        -- Extract P2 input
        local combined_input = joypad.get()
        local local_p2_input = {}
        
        for input, value in pairs(combined_input)
        do
            -- separate inputs from different players
            local prefix = string.sub(input,1,2)
            if prefix == "P2"
            then
                local_p2_input[string.sub(input,4,-1)] = BToN[value]
            end
        end

        -- Update class buffers with current frame state values
        P1:update(posXP1, posYP1, healthP1, superP1, stunP1, isStunnedP1, hitP1, beingThrownP1, nil)
        P2:update(posXP2, posYP2, healthP2, superP2, stunP2, isStunnedP2, hitP2, beingThrownP2, local_p2_input)

        -- Send the game state to model interface
        local gamestate = FormatState(P1,P2)
        local _, errmsg = Tcp:send(gamestate)
        if errmsg == "closed"
        then
            Tcp:close()
            Tcp = assert(socket.tcp())
            Tcp:connect(Host, Port)
            Tcp:send(gamestate)
        end
        
        -- Receive P1's input
        local raw_p1_input, err
        if Desynced then
            -- very naive way of emptying out the socket so that we dont get a desync between client and server
            Tcp:settimeout(0)
            local partial
            _, err, partial = Tcp:receive(100)
            if err == 'timeout' and partial ~= nil 
            then
                raw_p1_input = string.sub(partial, #partial-23, #partial)
                err = nil
            else
                Tcp:settimeout(Timeout)
                raw_p1_input, err = Tcp:receive('*l')
            end
            Tcp:settimeout(Timeout)
            Desynced = false
        else
            raw_p1_input, err = Tcp:receive('*l')
        end

        local split_p1_input
        -- if timeout repeat last input else get from server response
        if err == "timeout"
        then
            split_p1_input = P1.previousInput
            Desynced = true
        else
            split_p1_input = Split(raw_p1_input, ',')
        end
        local button_order = {'Left','Up','Right','Down','Weak Punch','Medium Punch','Strong Punch','Weak Kick','Medium Kick','Strong Kick','Start','Coin'}

        -- format and set P1's input in the game
        for i, button_name in ipairs(button_order)
        do
            -- change only p1's inputs
            local prefix = 'P1 '
            combined_input[prefix .. button_name] = SToB[split_p1_input[i]]
        end

        --{P2 Right=true, P2 Medium Punch=false, Service=false, P2 Coin=false, P1 Coin=false, P1 Down=true, P1 Strong Punch=false, P2 Weak Punch=false, P1 Weak Punch=false, P1 Medium Punch=false, P1 Start=false, P1 Medium Kick=false, P1 Right=false, P2 Up=false, P1 Strong Kick=false, Diagnostic=false, Region=1, P2 Down=false, P2 Left=false, P1 Left=true, P2 Medium Kick=false, Fake Dip=0, P2 Strong Punch=false, P1 Weak Kick=false, P2 Weak Kick=false, P1 Up=false, P2 Strong Kick=false, P2 Start=false, Reset=false}
        joypad.set(combined_input)

        -- Save 
        P1.previousInput = P1.inputs[#P1.inputs] or Split('0,0,0,0,0,0,0,0,0,0,0,0', ',')
        table.insert(P1.inputs, P1.inputs)

        -- Format everything and write to file
        if Frame_counter % Buff_size == 0
        then
            -- empty out the buffer classes so it doesnt slow everything down
            P1:wipe()
            P2:wipe()
        end

        -- Increase homemade framecounter
        Frame_counter = Frame_counter + 1

    elseif in_match == 6 and P1.posX[1] ~= nil -- if round has just ended and classes are not empty just write them to file
    then
        -- make sure that the player that lost has their health reduced to 0
        local finalHealthP1, finalHealthP2 = P1.health[#P1.health], P2.health[#P2.health]
        if finalHealthP1 < finalHealthP2
        then
            finalHealthP1 = 0
            hitP1 = 1
            hitP2 = 0
        elseif finalHealthP1 == finalHealthP2
        then
            finalHealthP1, finalHealthP2 = 0, 0
            hitP1, hitP2 = 1, 1
        else
            finalHealthP2 = 0
            hitP2 = 1
            hitP1 = 0
        end

        -- update the classes one last time for the end of match result (inputs are the same as previous frame for convenience)
        P1:update(P1.posX[#P1.posX], P1.posY[#P1.posY], finalHealthP1, P1.super[#P1.super], P1.stun[#P1.stun], P1.isStunned[#P1.isStunned], hitP1, P1.thrown[#P1.thrown], P1.inputs[#P1.inputs])
        P2:update(P2.posX[#P2.posX], P2.posY[#P2.posY], finalHealthP2, P2.super[#P2.super], P2.stun[#P2.stun], P2.isStunned[#P2.isStunned], hitP2, P2.thrown[#P2.thrown], P2.inputs[#P2.inputs])

        -- Reset global round specific variables
        P1:wipe()
        P2:wipe()
        P1.maxSuperBar = 0
        P2.maxSuperBar = 0
        P1.previousStun = 0
        P2.previousStun = 0
        P1.previousHealth = 161
        P2.previousHealth = 161
        P1.previousInput = Split('0,0,0,0,0,0,0,0,0,0,0,0', ',')
        P2.previousInput = Split('0,0,0,0,0,0,0,0,0,0,0,0', ',')
        StunnedP1, StunnedP2 = false, false
        CanRecoverFromStunP1, CanRecoverFromStunP2 = false, false
        HitStateP1, HitStateP2 = nil, nil

        RoundNumber = RoundNumber + 1
        Frame_counter = 1
    else
        return nil
    end
end

-- Establish TCP socket connection
Tcp = assert(socket.tcp())
Tcp:connect(Host, Port)
Tcp:settimeout(Timeout)
emu.registerbefore(GameInterface)
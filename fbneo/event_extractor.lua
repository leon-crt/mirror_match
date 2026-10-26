require("lua_libs/draw")
require("lua_libs/input_history")
require("lua_libs/gamestate")

package.path = "lua_libs/luasocket/?.lua;"
package.cpath = "lua_libs/socket/core.dll;" .. "lua_libs/mime/core.dll;"
local socket = require('socket')

frame_number = 1
Buff_size = 100
StunnedP1, StunnedP2 = false, false
CanRecoverFromStunP1, CanRecoverFromStunP2 = false, false
HitStateP1, HitStateP2 = nil, nil
RoundNumber = 0
Timeout = 3
Desynced = false
Screen_width = 383
Input_history_enabled = false
Control_both_characters = false
ChipDmgTolerance = 2
Host = "127.0.0.1"

ButtonsP1 = {'P1 Left','P1 Up','P1 Right','P1 Down','P1 Weak Punch','P1 Medium Punch','P1 Strong Punch','P1 Weak Kick','P1 Medium Kick','P1 Strong Kick','P1 Start','P1 Coin'}
ButtonsP2 = {'P2 Left','P2 Up','P2 Right','P2 Down','P2 Weak Punch','P2 Medium Punch','P2 Strong Punch','P2 Weak Kick','P2 Medium Kick','P2 Strong Kick','P2 Start','P2 Coin'}
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
    if (health < (player.previousHealth - ChipDmgTolerance))
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

function FormatState(p1, p2, extraInfo)
    local p1_data = string.format("%d,%d,%d,%d,%d,%d,%d,%d", p1.posX[#p1.posX], p1.posY[#p1.posY], p1.health[#p1.health], p1.super[#p1.super], p1.stun[#p1.stun], p1.isStunned[#p1.isStunned], p1.hit[#p1.hit], p1.thrown[#p1.thrown])
    local p2_data = string.format("%d,%d,%d,%d,%d,%d,%d,%d", p2.posX[#p2.posX], p2.posY[#p2.posY], p2.health[#p2.health], p2.super[#p2.super], p2.stun[#p2.stun], p2.isStunned[#p2.isStunned], p2.hit[#p2.hit], p2.thrown[#p2.thrown])
    local p2_inp = p2.inputs[#p2.inputs]
    local p2_inputs = string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", p2_inp["Left"] or 0, p2_inp["Up"] or 0, p2_inp["Right"] or 0, p2_inp["Down"] or 0, p2_inp["Weak Punch"] or 0, p2_inp["Medium Punch"] or 0, p2_inp["Strong Punch"] or 0, p2_inp["Weak Kick"] or 0, p2_inp["Medium Kick"] or 0, p2_inp["Strong Kick"] or 0, p2_inp["Start"] or 0, p2_inp["Coin"] or 0)

    -- add padding to make every message the same length
    local raw_string = string.format("%s,%s,%s,", p1_data, p2_data, p2_inputs)
    -- add extra info if present
    if extraInfo ~= nil then
        raw_string = raw_string .. extraInfo .. ','
    end
    
    local padding_len = 100 - #raw_string
    
    local padded_string = string.format("%s%s",raw_string, string.rep('#', padding_len))
    return padded_string, padding_len
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
    local currentActionP1, actionGroupP1
    -- Get current game phase
    local in_match = memory.readbyte(0x020154A7)  -- 1 = match intro, 2 = after round start, 9 = character select, 6 = end of round, 8 = transition between rounds
    if in_match == 1 
    then
        if P1.characterId == "" 
        then
            -- Extract P1 and P2 character and super IDs
            P1.characterId, P2.characterId = memory.readbyte(0x02011387), memory.readbyte(0x02011388)
            P1.superId, P2.superId = memory.readbyte(0x0201138B), memory.readbyte(0x0201138C)
            P1.superBarLength, P2.superBarLength = memory.readbyte(0x020695B3), memory.readbyte(0x020695DF)
        end
    end
    if in_match == 2 -- after round start
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
        currentActionP1, actionGroupP1 = tostring(memory.readbyte(0x02068E75)), memory.readbyte(0x02068E73)
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

        -- if testing, we need to add the action variables to the gamestate
        local extraInfo = nil
        extraInfo = string.format("%s,%s", currentActionP1, actionGroupP1)


        -- Send the game state to model interface
        local gamestate = FormatState(P1,P2, extraInfo)
        local _, errmsg = Client:send(gamestate)

        local continue_signal = false
        Client:settimeout(0)
        -- do busy waiting until the handler steps again
        while continue_signal == false do
            local msg = Client:receive("*l")
            if msg == "continue" then
                continue_signal = true
            end
        end

        if Input_history_enabled
        then
            input_history_update(input_history[1], "P1", combined_input)
            input_history_update(input_history[2], "P2", combined_input)
        end

        -- Format everything
        if frame_number % Buff_size == 0
        then
            -- empty out the buffer classes so it doesnt slow everything down
            P1:wipe()
            P2:wipe()
        end

        -- draw input history
        if Input_history_enabled 
        then
            input_history_draw(input_history[1], 4, 49, false)
            input_history_draw(input_history[2], Screen_width - 4, 49, true)
        end

        -- Increase homemade framecounter
        frame_number = frame_number + 1

    elseif in_match == 6 and P1.posX[1] ~= nil -- if round has just ended and classes are not empty just write them to file
    then
        RoundNumber = RoundNumber + 1
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
        
        -- Handle the action info if testing
        local extraInfo = nil
        extraInfo = string.format("%s,%s", tostring(0),tostring(0))

        -- Send the last state to the handler
        local gamestate = FormatState(P1,P2, extraInfo)

        -- Reset global round specific variables
        P1:wipe()
        P2:wipe()
        P1.maxSuperBar = 0
        P2.maxSuperBar = 0
        P1.previousStun = 0
        P2.previousStun = 0
        P1.previousHealth = 161
        P2.previousHealth = 161
        StunnedP1, StunnedP2 = false, false
        CanRecoverFromStunP1, CanRecoverFromStunP2 = false, false
        HitStateP1, HitStateP2 = nil, nil
        frame_number = 1

        gamestate = gamestate:sub(1, #gamestate -1) .. 'R' -- To signal the end of the round    

        local _, errmsg = Client:send(gamestate)
        if errmsg == "closed"
        then
            Client:close()
            Client = Tcp:accept()
            Client:settimeout(Timeout)
            Client:send(gamestate)
        end
    else
        return nil
    end
    if frame_number % 50 == 0
    then
        collectgarbage("collect")
    end
end

-- Establish TCP socket connection
local port = os.getenv("EMU_PORT")

Tcp = assert(socket.tcp())
local success, err = Tcp:bind(Host, port)
if success ~= 1
then
    Tcp:close()
    error(err)
end
Tcp:listen(0)
Client = Tcp:accept()
Client:settimeout(Timeout)
local data = Client:receive('*l')

if data == 'turbo' then
    emu.speedmode("turbo")
end

emu.registerbefore(GameInterface)
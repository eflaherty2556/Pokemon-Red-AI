-- require 'pylist'

--main functions
function progress()
	final_reward = moneyReward()
	final_reward = final_reward + partyReward()
	final_reward = final_reward + pkmn1XPReward()
	final_reward = final_reward + overworldMovementReward()
	final_reward = final_reward + explorationReward()
	
	return final_reward
end

function done_check()
	-- finish once 2 pokemon are obtained
	if data.party_size == 2 then
		return true
	end
	return false
end


--reward functions
previous_money = 3000
previous_party_size = 0
previous_pkmn1_exp = 0

previous_xPos = 0
previous_yPos = 0
movement_counter = 0
movement_counter_limit = 90
min_movement_delta = 2

--Initalize table
visitedMaps = {}


function moneyReward()
	return (data.money - previous_money) * 100.0
end

function partyReward()
	return (data.party_size - previous_party_size) * 2000
end

function pkmn1XPReward()
	return (data.totalExpPkmn1 - previous_pkmn1_exp) * 1500.0
end

function overworldMovementReward()
	if (movementDeltaWithinRange(data.xPosOverworld, previous_xPos) or movementDeltaWithinRange(data.yPosOverworld, previous_yPos)) then
		previous_xPos = data.xPosOverworld
		previous_yPos = data.yPosOverworld
		movement_counter = 0
		final_reward = 5000
	elseif movement_counter > movement_counter_limit then
		final_reward = -15 + (-15 * (movement_counter - movement_counter_limit) * 0.01)
		movement_counter = movement_counter + 1
		
	else
		movement_counter = movement_counter + 1
	end

	return final_reward
end

function movementDeltaWithinRange(current_pos, last_pos)
	pos_delta = math.abs(current_pos - last_pos)

	return pos_delta > min_movement_delta
end

function explorationReward()
	final_reward = 0
	if setContains(visitedMaps, data.mapID) then
		print("Exploring a new map!")
		addToSet(visitedMaps, data.mapID)
		final_reward = 50000
	end

	return final_reward
end


--list helper functions

function addToSet(set, key)
    set[key] = true
end

function removeFromSet(set, key)
    set[key] = nil
end

function setContains(set, key)
    return set[key] ~= nil
end

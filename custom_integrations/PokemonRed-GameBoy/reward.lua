previous_money = 3000
previous_party_size = 0
previous_pkmn1_exp = 0
function progress()
	return (((data.money - previous_money) * 0.25) + ((data.party_size - previous_party_size) * 100) + ((data.totalExpPkmn1 - previous_pkmn1_exp) * 1.0))
end

function done_check()
	-- finish once 6 pokemon are obtained
	if data.party_size == 6 then
		return true
	end
	return false
end
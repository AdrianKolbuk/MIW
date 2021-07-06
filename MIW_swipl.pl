mieszkanie(skrzynia).

platnerz(zbroja, 100).
platnerz(tarcza, 40).

zbrojmistrz(miecz, 50).
zbrojmistrz(mlot, 60).
zbrojmistrz(sztylet, 30).

alchemik(mikstura_zycia, 15, 10).
alchemik(mikstura_sily, 15, 10).

handlarz(zelazo, 5).
handlarz(ziola, 5).

rynek([platnerz, zbrojmistrz, alchemik]).

:- dynamic plecak/1.
:- dynamic portfel/1.
:- dynamic gdzie/1.
:- dynamic sprzedajRzecz/1.
:- dynamic memberchk/2.
plecak([zbroja, tarcza]).
portfel(200).
gdzie(mieszkanie).

idz(Y) :- gdzie(X), retract(gdzie(X)), assert(gdzie(Y)).
write_list([Head|Tail]) :- write(Head), nl, write_list(Tail).
wypiszplecak :- plecak(X), write_list(X).
dodaj(Y) :- plecak(X), append(X, [Y], Z), retractall(plecak(_)), assert(plecak(Z)), write(Z).
usun(Y) :- plecak(X), delete(X, Y, Z), retractall(plecak(_)), assert(plecak(Z)), write(Z).

kupPancerz(Rzecz) :- gdzie(X), platnerz(Rzecz, Cena), portfel(Pieniadze),
    X = 'rynek', Pieniadze>=Cena ->
    Pieniadzetmp is Pieniadze-Cena, retract(portfel(Pieniadze)), assert(portfel(Pieniadzetmp)),
    dodaj(Rzecz), write(' kupiono '), write(Rzecz)
    ;  
    write('Nie jestes na rynku lub przedmiot, ktory chcesz kupic jest za drogi').
	
kupBron(Rzecz) :- gdzie(X), zbrojmistrz(Rzecz, Cena), portfel(Pieniadze),
    X = 'rynek', Pieniadze>=Cena ->
    Pieniadzetmp is Pieniadze-Cena, retract(portfel(Pieniadze)), assert(portfel(Pieniadzetmp)),
    dodaj(Rzecz), write(' kupiono '), write(Rzecz)
    ;  
    write('Nie jestes na rynku lub przedmiot, ktory chcesz kupic jest za drogi').
    
sprzedajPancerz(Rzecz) :- platnerz(Rzecz, Cena), portfel(Pieniadze), plecak(Y),  
    member(Rzecz, Y) ->
    Pieniadzetmp is Pieniadze+Cena, retract(portfel(Pieniadze)), assert(portfel(Pieniadzetmp)),
    usun(Rzecz), write('sprzedano '), write(Rzecz)
    ;
	write('nie masz takiej rzeczy w plecaku').
	
sprzedajBron(Rzecz) :- zbrojmistrz(Rzecz, Cena), portfel(Pieniadze), plecak(Y),  
    member(Rzecz, Y) ->
    Pieniadzetmp is Pieniadze+Cena, retract(portfel(Pieniadze)), assert(portfel(Pieniadzetmp)),
    usun(Rzecz), write('sprzedano '), write(Rzecz)
    ;
	write('nie masz takiej rzeczy w plecaku').
	






